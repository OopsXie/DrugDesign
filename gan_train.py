"""
GAN Training Script for DrugGAN-MSM

This script implements the training procedure for the DrugGAN-MSM conditional GAN model
as described in Chapter 3 of the thesis. It trains the Generator and MultiTaskDiscriminator
alternately using the joint loss function:

    L = λ1 * L_GAN + λ2 * L_bio + λ3 * L_chem

Where:
- L_GAN: Adversarial loss (Formula 3-14)
- L_bio: Biology property loss (Formula 3-15)
- L_chem: Chemical validity loss (Formula 3-16)

Author: DrugGAN-MSM Team
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, AllChem, DataStructs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from project
from model.DrugGAN_MSM import Generator, MultiTaskDiscriminator, ProteinEncoder, SMILESEncoder
from utils.baseline import prepareDataset, loadConfig, MyDataset, fetchIndices, splitSmi
from utils.sascorer import calculateScore as compute_sa_score
from utils.log import prepareFolder, trainingVis

from model.train_utils import GANLoss, BiologyPropertyLoss, compute_mol_properties, set_seed

# ============================================================================
# Enhanced Loss (gan_train-specific, with fingerprint similarity scoring)
# ============================================================================
class EnhancedChemicalValidityLoss(nn.Module):
    """
    Enhanced chemical validity loss with composite scoring (uniqueness, fingerprint similarity, SA normalization).

    Per-molecule score: S_i = (validity + uniqueness + sa_norm + sim_score) / 4.0
    L_chem = -Σ S_i (negated because we minimize)
    """
    
    def __init__(self, training_smiles: Optional[List[str]] = None):
        super().__init__()
        self.training_fps = []
        if training_smiles is not None:
            for smi in training_smiles[:100]:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    self.training_fps.append(fp)
    
    def _compute_fp_similarity(self, mol: Chem.Mol) -> float:
        """Compute average Tanimoto similarity to training set molecules."""
        if not self.training_fps:
            return 0.5
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        similarities = [DataStructs.TanimotoSimilarity(fp, train_fp) for train_fp in self.training_fps]
        return float(np.mean(similarities))
    
    def forward(
        self,
        valid_mask: List[int],
        generated_smiles: List[str],
        sa_scores: List[float],
        batch_size: int,
        training_smiles: Optional[Set[str]] = None
    ) -> torch.Tensor:
        """
        Compute chemical validity loss with composite score.
        
        Args:
            valid_mask: Binary mask indicating valid molecules (1=valid, 0=invalid)
            generated_smiles: List of generated SMILES strings
            sa_scores: SA scores for each molecule (1-10 scale)
            batch_size: Size of the batch
            training_smiles: Optional set of training SMILES for novelty check
            
        Returns:
            Scalar chemical validity loss
        """
        if batch_size == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        
        total_score = 0.0
        
        for i in range(batch_size):
            validity = 1.0 if valid_mask[i] == 1 else 0.0
            
            if validity == 0:
                total_score += 0.0
                continue
            
            smi = generated_smiles[i]
            
            uniqueness_count = generated_smiles.count(smi)
            uniqueness = 1.0 / uniqueness_count if uniqueness_count > 0 else 0.0
            
            sa_norm = 1.0 - (sa_scores[i] / 10.0)
            sa_norm = max(0.0, min(1.0, sa_norm))
            
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    sim_score = self._compute_fp_similarity(mol)
                else:
                    sim_score = 0.0
            except Exception:
                sim_score = 0.0
            
            S_i = (validity + uniqueness + sa_norm + sim_score) / 4.0
            total_score += S_i
        
        loss = -total_score
        return torch.tensor(loss, dtype=torch.float32)


class JointLoss(nn.Module):
    """
    Joint loss combining all components (Formula 3-13, 3-17).
    
    L = λ1 * L_GAN + λ2 * L_bio + λ3 * L_chem
    
    Default weights: λ1=1.0, λ2=0.5, λ3=0.5
    """
    
    def __init__(self, lambda_gan: float = 1.0, lambda_bio: float = 0.5, lambda_chem: float = 0.5):
        super(JointLoss, self).__init__()
        self.lambda_gan = lambda_gan
        self.lambda_bio = lambda_bio
        self.lambda_chem = lambda_chem
        
        self.gan_loss = GANLoss()
        self.bio_loss = BiologyPropertyLoss()
        self.chem_loss = EnhancedChemicalValidityLoss()
    
    def discriminator_loss(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
        pred_qed_real: torch.Tensor,
        pred_sa_real: torch.Tensor,
        pred_logp_real: torch.Tensor,
        pred_affinity_real: torch.Tensor,
        true_qed: List[float],
        true_sa: List[float],
        true_logp: List[float],
        true_affinity: List[float]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total discriminator loss.
        
        Args:
            d_real: Discriminator output for real samples
            d_fake: Discriminator output for fake samples
            pred_*_real: Property predictions for real samples
            true_*: Ground truth properties for real samples
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """

        gan_loss = self.gan_loss.forward_discriminator(d_real, d_fake)
        

        bio_loss_val = self.bio_loss(
            pred_qed_real, pred_sa_real, pred_logp_real, pred_affinity_real,
            true_qed, true_sa, true_logp, true_affinity
        )
        

        total_loss = self.lambda_gan * gan_loss + self.lambda_bio * bio_loss_val
        
        loss_dict = {
            'gan_loss': gan_loss.item(),
            'bio_loss': bio_loss_val.item(),
            'chem_loss': 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def generator_loss(
        self,
        d_fake: torch.Tensor,
        pred_qed_fake: torch.Tensor,
        pred_sa_fake: torch.Tensor,
        pred_logp_fake: torch.Tensor,
        pred_affinity_fake: torch.Tensor,
        true_qed: List[float],
        true_sa: List[float],
        true_logp: List[float],
        true_affinity: List[float],
        valid_mask: List[int],
        batch_size: int,
        generated_smiles: List[str],
        sa_scores: List[float]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total generator loss.
        
        Args:
            d_fake: Discriminator output for fake samples
            pred_*_fake: Property predictions for fake samples
            true_*: Target properties (we want generated molecules to have good properties)
            valid_mask: Validity mask for generated molecules
            batch_size: Batch size
            generated_smiles: List of generated SMILES strings
            sa_scores: SA scores for generated molecules
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """

        gan_loss = self.gan_loss.forward_generator(d_fake)
        


        ideal_qed = [1.0] * batch_size
        ideal_sa = [1.0] * batch_size
        ideal_logp = [2.0] * batch_size
        ideal_affinity = [-10.0] * batch_size
        
        bio_loss_val = self.bio_loss(
            pred_qed_fake, pred_sa_fake, pred_logp_fake, pred_affinity_fake,
            ideal_qed, ideal_sa, ideal_logp, ideal_affinity
        )
        

        chem_loss_val = self.chem_loss(valid_mask, generated_smiles, sa_scores, batch_size)
        

        total_loss = (
            self.lambda_gan * gan_loss +
            self.lambda_bio * bio_loss_val +
            self.lambda_chem * chem_loss_val
        )
        
        loss_dict = {
            'gan_loss': gan_loss.item(),
            'bio_loss': bio_loss_val.item(),
            'chem_loss': chem_loss_val.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# Dataset Class for GAN Training
# ============================================================================

class GANDataset(torch.utils.data.Dataset):
    """
    Dataset for GAN training that includes raw SMILES strings for property computation.
    
    Extends the baseline dataset to provide access to raw SMILES strings
    needed for RDKit property computation.
    """
    
    def __init__(self, data: Tuple, raw_smiles_list: List[str]):
        """
        Initialize dataset.
        
        Args:
            data: Tuple of (proIndices, smiIndices, labelIndices, proMask, smiMask)
            raw_smiles_list: List of raw SMILES strings for property computation
        """
        proIndices, smiIndices, labelIndices, proMask, smiMask = data
        self._len = len(proIndices)
        self.x = proIndices
        self.y = smiIndices
        self.label = labelIndices
        self.proMask = proMask
        self.smiMask = smiMask
        self.raw_smiles = raw_smiles_list
    
    def __getitem__(self, idx: int):
        proMask = [1.0] * self.proMask[idx] + [0.0] * (len(self.x[idx]) - self.proMask[idx])
        smiMask = [1.0] * self.smiMask[idx] + [0.0] * (len(self.label[idx]) - self.smiMask[idx])
        
        return (
            self.x[idx],
            self.y[idx],
            self.label[idx],
            np.array(proMask).astype(int),
            np.array(smiMask).astype(int),
            self.raw_smiles[idx]
        )
    
    def __len__(self):
        return self._len


def prepareGANDataset(config, orign: str, data_dir: str = './data') -> Tuple[List, List[str]]:
    """
    Prepare dataset for GAN training including raw SMILES strings.
    
    Args:
        config: Configuration object with vocabularies and max lengths
        orign: Dataset split name ('train' or 'valid')
        data_dir: Directory containing train-val-data.tsv and train-val-split.json
        
    Returns:
        Tuple of (encoded_data, raw_smiles_list)
    """
    import json
    
    split_path = os.path.join(data_dir, 'train-val-split.json')
    data_path = os.path.join(data_dir, 'train-val-data.tsv')
    
    with open(split_path, 'r') as f:
        data_config = json.load(f)
    
    slices = data_config[orign]
    data = pd.read_csv(data_path, sep='\t')
    data = data.loc[slices]
    

    raw_smiles_list = data['smiles'].tolist()
    

    smiArr = data['smiles'].apply(splitSmi).tolist()
    proArr = data['protein'].apply(list).tolist()
    

    smiIndices, labelIndices, smiMask = fetchIndices(smiArr, config.smiVoc, config.smiMaxLen)
    proIndices, _, proMask = fetchIndices(proArr, config.proVoc, config.proMaxLen)
    
    return (proIndices, smiIndices, labelIndices, proMask, smiMask), raw_smiles_list


# ============================================================================
# Training Functions
# ============================================================================

def train_discriminator_step(
    batch: Tuple,
    protein_encoder: ProteinEncoder,
    smi_encoder: SMILESEncoder,
    generator: Generator,
    discriminator: MultiTaskDiscriminator,
    joint_loss: JointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    teacher_forcing: bool = True
) -> Dict[str, float]:
    """
    Perform one discriminator update step.
    
    Args:
        batch: Data batch (protein, smiles, label, pro_mask, smi_mask, raw_smiles)
        protein_encoder: Protein encoder module
        smi_encoder: SMILES encoder module
        generator: Generator module
        discriminator: Discriminator module
        joint_loss: Joint loss function
        optimizer: Discriminator optimizer
        device: Device to run on
        smiVoc: SMILES vocabulary
        teacher_forcing: Whether to use teacher forcing for generation
        
    Returns:
        Dictionary of loss values
    """
    protein, smiles, label, pro_mask, smi_mask, raw_smiles = batch
    

    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    label = torch.as_tensor(label).to(device)
    
    batch_size = protein.size(0)
    

    protein_output = protein_encoder(protein, apply_mlm=False)
    protein_features = protein_output['protein_features']
    

    real_mol_output = smi_encoder(smiles)
    real_mol_features = real_mol_output['token_features']
    

    disc_real_output = discriminator(real_mol_features, protein_features)
    

    if teacher_forcing:


        tgt_mask = torch.triu(
            torch.ones(smiles.size(1), smiles.size(1), device=device),
            diagonal=1
        ).masked_fill(torch.ones(smiles.size(1), smiles.size(1), device=device) == 1, float('-inf'))
        
        gen_logits = generator(smiles, protein_features)
        
        fake_smiles = gen_logits.argmax(dim=-1)
    else:

        start_token = smiVoc.index('&')
        fake_smiles = generator.generate(
            protein_features,
            max_len=smiles.size(1),
            start_token=start_token
        )
    

    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    

    disc_fake_output = discriminator(fake_mol_features, protein_features)
    

    real_props = compute_mol_properties(raw_smiles)
    

    loss, loss_dict = joint_loss.discriminator_loss(
        d_real=disc_real_output['real_fake'],
        d_fake=disc_fake_output['real_fake'],
        pred_qed_real=disc_real_output['qed'],
        pred_sa_real=disc_real_output['sa'],
        pred_logp_real=disc_real_output['logp'],
        pred_affinity_real=disc_real_output['affinity'],
        true_qed=real_props['qed'],
        true_sa=real_props['sa'],
        true_logp=real_props['logp'],
        true_affinity=real_props['affinity']
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(protein_encoder.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(smi_encoder.parameters(), 1.0)
    optimizer.step()
    
    return loss_dict


def train_generator_step(
    batch: Tuple,
    protein_encoder: ProteinEncoder,
    smi_encoder: SMILESEncoder,
    generator: Generator,
    discriminator: MultiTaskDiscriminator,
    joint_loss: JointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    teacher_forcing: bool = True
) -> Dict[str, float]:
    """
    Perform one generator update step.
    
    Args:
        batch: Data batch (protein, smiles, label, pro_mask, smi_mask, raw_smiles)
        protein_encoder: Protein encoder module
        smi_encoder: SMILES encoder module
        generator: Generator module
        discriminator: Discriminator module
        joint_loss: Joint loss function
        optimizer: Generator optimizer
        device: Device to run on
        smiVoc: SMILES vocabulary
        teacher_forcing: Whether to use teacher forcing
        
    Returns:
        Dictionary of loss values
    """
    protein, smiles, label, pro_mask, smi_mask, raw_smiles = batch
    

    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    
    batch_size = protein.size(0)
    

    protein_output = protein_encoder(protein, apply_mlm=False)
    protein_features = protein_output['protein_features']
    

    if teacher_forcing:

        tgt_mask = torch.triu(
            torch.ones(smiles.size(1), smiles.size(1), device=device),
            diagonal=1
        ).masked_fill(torch.ones(smiles.size(1), smiles.size(1), device=device) == 1, float('-inf'))
        
        gen_logits = generator(smiles, protein_features)
        
        fake_smiles = gen_logits.argmax(dim=-1)
    else:

        start_token = smiVoc.index('&')
        fake_smiles = generator.generate(
            protein_features,
            max_len=smiles.size(1),
            start_token=start_token
        )
    

    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    

    disc_fake_output = discriminator(fake_mol_features, protein_features)
    


    fake_smiles_list = []
    for i in range(batch_size):
        tokens = []
        for idx in fake_smiles[i]:
            token = smiVoc[idx.item()]
            if token in ['&', '$', '^']:
                if token == '$':
                    break
                continue
            tokens.append(token)
        smi = ''.join(tokens)
        fake_smiles_list.append(smi)
    

    valid_mask = []
    sa_scores = []
    for smi in fake_smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mask.append(1)
                sa_scores.append(compute_sa_score(mol))
            else:
                valid_mask.append(0)
                sa_scores.append(10.0)
        except:
            valid_mask.append(0)
            sa_scores.append(10.0)
    

    loss, loss_dict = joint_loss.generator_loss(
        d_fake=disc_fake_output['real_fake'],
        pred_qed_fake=disc_fake_output['qed'],
        pred_sa_fake=disc_fake_output['sa'],
        pred_logp_fake=disc_fake_output['logp'],
        pred_affinity_fake=disc_fake_output['affinity'],
        true_qed=[],
        true_sa=[],
        true_logp=[],
        true_affinity=[],
        valid_mask=valid_mask,
        batch_size=batch_size,
        generated_smiles=fake_smiles_list,
        sa_scores=sa_scores
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
    optimizer.step()
    
    
    validity_rate = sum(valid_mask) / max(batch_size, 1)
    loss_dict['validity_rate'] = validity_rate
    
    return loss_dict


def train_epoch(
    generator: Generator,
    discriminator: MultiTaskDiscriminator,
    protein_encoder: ProteinEncoder,
    smi_encoder: SMILESEncoder,
    train_loader: DataLoader,
    joint_loss: JointLoss,
    gen_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    teacher_forcing: bool = True,
    n_critic: int = 1
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        protein_encoder: Protein encoder
        smi_encoder: SMILES encoder
        train_loader: Training data loader
        joint_loss: Joint loss function
        gen_optimizer: Generator optimizer
        disc_optimizer: Discriminator optimizer
        device: Device to run on
        smiVoc: SMILES vocabulary
        teacher_forcing: Whether to use teacher forcing
        n_critic: Number of discriminator updates per generator update
        
    Returns:
        Dictionary of average loss values for the epoch
    """
    generator.train()
    discriminator.train()
    protein_encoder.train()
    smi_encoder.train()
    
    epoch_losses = {
        'gen_total': 0.0,
        'gen_gan': 0.0,
        'gen_bio': 0.0,
        'gen_chem': 0.0,
        'disc_total': 0.0,
        'disc_gan': 0.0,
        'disc_bio': 0.0,
        'validity_rate': 0.0
    }
    
    num_batches = len(train_loader)
    
    for batch in tqdm(train_loader, desc='Training'):

        batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
        

        for _ in range(n_critic):
            disc_losses = train_discriminator_step(
                batch, protein_encoder, smi_encoder, generator, discriminator,
                joint_loss, disc_optimizer, device, smiVoc, teacher_forcing
            )
            epoch_losses['disc_total'] += disc_losses['total_loss']
            epoch_losses['disc_gan'] += disc_losses['gan_loss']
            epoch_losses['disc_bio'] += disc_losses['bio_loss']
        

        gen_losses = train_generator_step(
            batch, protein_encoder, smi_encoder, generator, discriminator,
            joint_loss, gen_optimizer, device, smiVoc, teacher_forcing
        )
        epoch_losses['gen_total'] += gen_losses['total_loss']
        epoch_losses['gen_gan'] += gen_losses['gan_loss']
        epoch_losses['gen_bio'] += gen_losses['bio_loss']
        epoch_losses['gen_chem'] += gen_losses['chem_loss']
        epoch_losses['validity_rate'] += gen_losses.get('validity_rate', 0.0)
    

    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


@torch.no_grad()
def validate_epoch(
    generator: Generator,
    discriminator: MultiTaskDiscriminator,
    protein_encoder: ProteinEncoder,
    smi_encoder: SMILESEncoder,
    val_loader: DataLoader,
    joint_loss: JointLoss,
    device: torch.device,
    smiVoc: List[str],
    teacher_forcing: bool = True
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        protein_encoder: Protein encoder
        smi_encoder: SMILES encoder
        val_loader: Validation data loader
        joint_loss: Joint loss function
        device: Device to run on
        smiVoc: SMILES vocabulary
        teacher_forcing: Whether to use teacher forcing
        
    Returns:
        Dictionary of average loss values for validation
    """
    generator.eval()
    discriminator.eval()
    protein_encoder.eval()
    smi_encoder.eval()
    
    val_losses = {
        'gen_total': 0.0,
        'gen_gan': 0.0,
        'gen_bio': 0.0,
        'gen_chem': 0.0,
        'disc_total': 0.0,
        'disc_gan': 0.0,
        'disc_bio': 0.0,
        'validity_rate': 0.0
    }
    
    num_batches = len(val_loader)
    
    for batch in tqdm(val_loader, desc='Validating'):
        batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
        protein, smiles, label, pro_mask, smi_mask, raw_smiles = batch
        batch_size = protein.size(0)
        

        protein_output = protein_encoder(protein, apply_mlm=False)
        protein_features = protein_output['protein_features']
        

        if teacher_forcing:
            gen_logits = generator(smiles, protein_features)
            fake_smiles = gen_logits.argmax(dim=-1)
        else:
            start_token = smiVoc.index('&')
            fake_smiles = generator.generate(
                protein_features,
                max_len=smiles.size(1),
                start_token=start_token
            )
        

        real_mol_output = smi_encoder(smiles)
        real_mol_features = real_mol_output['token_features']
        
        fake_mol_output = smi_encoder(fake_smiles)
        fake_mol_features = fake_mol_output['token_features']
        

        disc_real_output = discriminator(real_mol_features, protein_features)
        disc_fake_output = discriminator(fake_mol_features, protein_features)
        

        real_props = compute_mol_properties(raw_smiles)
        

        disc_loss, _ = joint_loss.discriminator_loss(
            d_real=disc_real_output['real_fake'],
            d_fake=disc_fake_output['real_fake'],
            pred_qed_real=disc_real_output['qed'],
            pred_sa_real=disc_real_output['sa'],
            pred_logp_real=disc_real_output['logp'],
            pred_affinity_real=disc_real_output['affinity'],
            true_qed=real_props['qed'],
            true_sa=real_props['sa'],
            true_logp=real_props['logp'],
            true_affinity=real_props['affinity']
        )
        

        fake_smiles_list = []
        for i in range(batch_size):
            tokens = []
            for idx in fake_smiles[i]:
                token = smiVoc[idx.item()]
                if token in ['&', '$', '^']:
                    if token == '$':
                        break
                    continue
                tokens.append(token)
            smi = ''.join(tokens)
            fake_smiles_list.append(smi)
        
        valid_mask = []
        sa_scores = []
        for smi in fake_smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_mask.append(1)
                    sa_scores.append(compute_sa_score(mol))
                else:
                    valid_mask.append(0)
                    sa_scores.append(10.0)
            except:
                valid_mask.append(0)
                sa_scores.append(10.0)
        
        gen_loss, gen_loss_dict = joint_loss.generator_loss(
            d_fake=disc_fake_output['real_fake'],
            pred_qed_fake=disc_fake_output['qed'],
            pred_sa_fake=disc_fake_output['sa'],
            pred_logp_fake=disc_fake_output['logp'],
            pred_affinity_fake=disc_fake_output['affinity'],
            true_qed=[],
            true_sa=[],
            true_logp=[],
            true_affinity=[],
            valid_mask=valid_mask,
            batch_size=batch_size,
            generated_smiles=fake_smiles_list,
            sa_scores=sa_scores
        )
        

        val_losses['disc_total'] += disc_loss.item()
        val_losses['gen_total'] += gen_loss.item()
        val_losses['gen_gan'] += gen_loss_dict['gan_loss']
        val_losses['gen_bio'] += gen_loss_dict['bio_loss']
        val_losses['gen_chem'] += gen_loss_dict['chem_loss']
        val_losses['validity_rate'] += sum(valid_mask) / max(batch_size, 1)
    

    for key in val_losses:
        val_losses[key] /= num_batches
    
    return val_losses


def save_checkpoint(
    epoch: int,
    generator: Generator,
    discriminator: MultiTaskDiscriminator,
    protein_encoder: ProteinEncoder,
    smi_encoder: SMILESEncoder,
    gen_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    config: Dict[str, Any],
    output_dir: str
):
    """
    Save model checkpoint.
    
    Args:
        epoch: Current epoch number
        generator: Generator model
        discriminator: Discriminator model
        protein_encoder: Protein encoder
        smi_encoder: SMILES encoder
        gen_optimizer: Generator optimizer
        disc_optimizer: Discriminator optimizer
        config: Training configuration
        output_dir: Output directory
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'protein_encoder_state_dict': protein_encoder.state_dict(),
        'smi_encoder_state_dict': smi_encoder.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'config': config
    }
    
    checkpoint_path = os.path.join(output_dir, 'model', f'epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def plot_loss_curves(
    train_losses: List[Dict[str, float]],
    val_losses: List[Dict[str, float]],
    output_dir: str
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training loss dictionaries per epoch
        val_losses: List of validation loss dictionaries per epoch
        output_dir: Output directory for plots
    """
    epochs = range(1, len(train_losses) + 1)
    

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    

    ax = axes[0, 0]
    train_gen_total = [loss['gen_total'] for loss in train_losses]
    val_gen_total = [loss['gen_total'] for loss in val_losses]
    ax.plot(epochs, train_gen_total, 'b-', label='Train Gen Total')
    ax.plot(epochs, val_gen_total, 'r-', label='Val Gen Total')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Generator Total Loss')
    ax.legend()
    ax.grid(True)
    

    ax = axes[0, 1]
    train_disc_total = [loss['disc_total'] for loss in train_losses]
    val_disc_total = [loss['disc_total'] for loss in val_losses]
    ax.plot(epochs, train_disc_total, 'b-', label='Train Disc Total')
    ax.plot(epochs, val_disc_total, 'r-', label='Val Disc Total')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator Total Loss')
    ax.legend()
    ax.grid(True)
    

    ax = axes[1, 0]
    train_gen_gan = [loss['gen_gan'] for loss in train_losses]
    train_gen_bio = [loss['gen_bio'] for loss in train_losses]
    train_gen_chem = [loss['gen_chem'] for loss in train_losses]
    ax.plot(epochs, train_gen_gan, 'b-', label='GAN Loss')
    ax.plot(epochs, train_gen_bio, 'g-', label='Bio Loss')
    ax.plot(epochs, train_gen_chem, 'r-', label='Chem Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Generator Loss Components')
    ax.legend()
    ax.grid(True)
    

    ax = axes[1, 1]
    train_validity = [loss['validity_rate'] for loss in train_losses]
    val_validity = [loss['validity_rate'] for loss in val_losses]
    ax.plot(epochs, train_validity, 'b-', label='Train Validity')
    ax.plot(epochs, val_validity, 'r-', label='Val Validity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate')
    ax.set_title('Molecule Validity Rate')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logs', 'loss_curves.png'), dpi=150)
    plt.close()
    logger.info(f"Saved loss curves to {os.path.join(output_dir, 'logs', 'loss_curves.png')}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function."""

    parser = argparse.ArgumentParser(description='Train DrugGAN-MSM')
    

    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-critic', type=int, default=1, help='Number of critic updates per generator update')
    

    parser.add_argument('--d-model', type=int, default=512, help='Model hidden dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of protein encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=12, help='Number of generator decoder layers')
    parser.add_argument('--num-disc-layers', type=int, default=6, help='Number of discriminator encoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--noise-dim', type=int, default=128, help='Noise vector dimension')
    parser.add_argument('--mask-rate', type=float, default=0.15, help='MLM mask rate')
    

    parser.add_argument('--lambda-gan', type=float, default=1.0, help='GAN loss weight')
    parser.add_argument('--lambda-bio', type=float, default=0.5, help='Biology loss weight')
    parser.add_argument('--lambda-chem', type=float, default=0.5, help='Chemical validity loss weight')
    

    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--config-path', type=str, default='', help='Path to train-val-split.json (default: data-dir/train-val-split.json)')
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--output-dir', type=str, default='./experiments/druggan_msm', help='Output directory')
    parser.add_argument('--note', type=str, default='', help='Experiment note')
    parser.add_argument('--teacher-forcing', action='store_true', default=True, help='Use teacher forcing')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience epochs')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    

    logger.add(os.path.join(args.output_dir, 'logs', 'train.log'))
    logger.info(f"Training started at {datetime.now()}")
    logger.info(f"Arguments: {args}")
    

    config = loadConfig(args)
    config.batchSize = args.batch_size
    

    settings = {
        'remark': args.note,
        'smiVoc': config.smiVoc,
        'proVoc': config.proVoc,
        'smiMaxLen': config.smiMaxLen,
        'proMaxLen': config.proMaxLen,
        'smiPaddingIdx': config.smiVoc.index('^'),
        'proPaddingIdx': config.proVoc.index('^'),
        'smi_voc_len': len(config.smiVoc),
        'pro_voc_len': len(config.proVoc),
        'batchSize': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'd_model': args.d_model,
        'dim_feedforward': args.dim_feedforward,
        'num_encoder_layers': args.num_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'num_disc_layers': args.num_disc_layers,
        'nhead': args.nhead,
        'noise_dim': args.noise_dim,
        'mask_rate': args.mask_rate,
        'lambda_gan': args.lambda_gan,
        'lambda_bio': args.lambda_bio,
        'lambda_chem': args.lambda_chem,
        'n_critic': args.n_critic
    }
    
    with open(os.path.join(args.output_dir, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)
    logger.info("Saved training settings")
    

    logger.info("Preparing training data...")
    train_data, train_smiles = prepareGANDataset(config, 'train', args.data_dir)
    val_data, val_smiles = prepareGANDataset(config, 'valid', args.data_dir)
    
    train_dataset = GANDataset(train_data, train_smiles)
    val_dataset = GANDataset(val_data, val_smiles)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    

    logger.info("Initializing models...")
    
    protein_encoder = ProteinEncoder(
        pro_voc_len=len(config.proVoc),
        d_model=args.d_model,
        mask_rate=args.mask_rate
    )
    
    smi_encoder = SMILESEncoder(
        smi_voc_len=len(config.smiVoc),
        d_model=args.d_model
    )
    
    generator = Generator(
        smi_voc_len=len(config.smiVoc),
        d_model=args.d_model,
        noise_dim=args.noise_dim,
        num_layers=args.num_decoder_layers
    )
    
    discriminator = MultiTaskDiscriminator(
        d_model=args.d_model,
        hidden_dim=args.d_model // 2,
        protein_dim=756
    )
    

    protein_encoder = protein_encoder.to(device)
    smi_encoder = smi_encoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    

    gen_params = list(generator.parameters())
    disc_params = list(discriminator.parameters()) + list(protein_encoder.parameters()) + list(smi_encoder.parameters())
    
    gen_optimizer = optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_params, lr=args.lr, betas=(0.5, 0.999))
    
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=args.epochs, eta_min=1e-6)
    disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=args.epochs, eta_min=1e-6)
    

    joint_loss = JointLoss(
        lambda_gan=args.lambda_gan,
        lambda_bio=args.lambda_bio,
        lambda_chem=args.lambda_chem
    )
    

    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            protein_encoder.load_state_dict(checkpoint['protein_encoder_state_dict'])
            smi_encoder.load_state_dict(checkpoint['smi_encoder_state_dict'])
            gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    

    train_history = []
    val_history = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Setup device
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_losses = train_epoch(
            generator, discriminator, protein_encoder, smi_encoder,
            train_loader, joint_loss, gen_optimizer, disc_optimizer,
            device, config.smiVoc, teacher_forcing=args.teacher_forcing,
            n_critic=args.n_critic
        )
        
        # Validate
        val_losses = validate_epoch(
            generator, discriminator, protein_encoder, smi_encoder,
            val_loader, joint_loss, device, config.smiVoc,
            teacher_forcing=args.teacher_forcing
        )
        
        # Log
        logger.info(f"Train - Gen Total: {train_losses['gen_total']:.4f}, "
                   f"Disc Total: {train_losses['disc_total']:.4f}, "
                   f"Validity: {train_losses['validity_rate']:.4f}")
        logger.info(f"Val   - Gen Total: {val_losses['gen_total']:.4f}, "
                   f"Disc Total: {val_losses['disc_total']:.4f}, "
                   f"Validity: {val_losses['validity_rate']:.4f}")
        
        # History
        train_history.append(train_losses)
        val_history.append(val_losses)
        
        # Learning rate schedulers
        gen_scheduler.step()
        disc_scheduler.step()
        
        # Early stopping check
        val_gen_loss = val_losses['gen_total']
        if val_gen_loss < best_val_loss:
            best_val_loss = val_gen_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(
                epoch, generator, discriminator, protein_encoder, smi_encoder,
                gen_optimizer, disc_optimizer, settings, 
                os.path.join(args.output_dir, 'model_best')
            )
            logger.info(f"New best model saved (val_gen_loss={val_gen_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                epoch, generator, discriminator, protein_encoder, smi_encoder,
                gen_optimizer, disc_optimizer, settings, args.output_dir
            )
        
        # Plot
        plot_loss_curves(train_history, val_history, args.output_dir)
    

    save_checkpoint(
        args.epochs - 1, generator, discriminator, protein_encoder, smi_encoder,
        gen_optimizer, disc_optimizer, settings, args.output_dir
    )
    

    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logger.info(f"\nTraining completed in {total_time:.2f} hours")
    logger.info(f"Final model saved to {os.path.join(args.output_dir, 'model', f'epoch_{args.epochs - 1}.pt')}")


if __name__ == '__main__':
    main()