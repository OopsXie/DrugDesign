"""
Ablation Study Training Script for DrugGAN-MSM

This script runs ablation studies for Chapter 3 (DrugGAN-MSM) and Chapter 4 (KG-DrugGAN-MSM).
It imports shared components from model/train_utils.py and model/DrugGAN_MSM.py.

Author: DrugGAN-MSM Team
"""

import os
import sys
import json
import time
import argparse
import csv
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

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
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import shared utilities
from model.train_utils import (
    GANLoss,
    BiologyPropertyLoss,
    ChemicalValidityLoss,
    compute_mol_properties,
    compute_uniqueness,
    set_seed
)

# Import base model components
from model.DrugGAN_MSM import (
    ProteinEncoder,
    SMILESEncoder,
    Generator,
    MultiTaskDiscriminator,
    PositionalEncoding,
    ProteinEncoderNoMLM,
    ProteinEncoderNoSA,
    SMILESEncoderNoMLM,
    SimpleDiscriminator
)

# Import KG components
from model.KG_DrugGAN_MSM import (
    KGDrugGANModel,
    GatedFusion,
    KnowledgeProjector,
    KnowledgeConsistencyLoss,
    KGConditionBuilder
)
from kg.kg_dataset import KGDataset, build_kg_from_tsv
from kg.kg_transe import TransEModel, train_transe, aggregate_neighborhood
from utils.baseline import prepareDataset, loadConfig, MyDataset, fetchIndices, splitSmi


# ============================================================================
# SHARED LOSS WRAPPER (unique to ablation)
# ============================================================================

class KnowledgeConsistencyLossWrapper(nn.Module):
    """Wrapper for knowledge consistency loss using MSE."""
    
    def __init__(self):
        super(KnowledgeConsistencyLossWrapper, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, h_mol: torch.Tensor, h_kg: torch.Tensor) -> torch.Tensor:
        if h_mol.dim() > 2:
            h_mol = h_mol.mean(dim=1)
        return self.mse_loss(h_mol, h_kg)


class KGJointLoss(nn.Module):
    """
    Joint loss for KG-enhanced GAN training.
    
    Combines:
    - GAN adversarial loss
    - Biology property loss (QED, SA, logP)
    - Chemical validity loss
    - Knowledge consistency loss (KG mode only)
    """
    
    def __init__(self, lambda_gan: float = 1.0, alpha: float = 0.5, beta: float = 0.5,
                 lambda_bio: float = 0.3, lambda_chem: float = 0.2):
        super(KGJointLoss, self).__init__()
        self.lambda_gan = lambda_gan
        self.alpha = alpha
        self.beta = beta
        self.lambda_bio = lambda_bio
        self.lambda_chem = lambda_chem
        
        self.gan_loss = GANLoss()
        self.bio_loss = BiologyPropertyLoss()
        self.chem_loss = ChemicalValidityLoss()
        self.kg_loss = KnowledgeConsistencyLossWrapper()
    
    def discriminator_loss(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
        pred_qed_real: torch.Tensor,
        pred_sa_real: torch.Tensor,
        pred_logp_real: torch.Tensor,
        true_qed: List[float],
        true_sa: List[float],
        true_logp: List[float],
        use_multi_task: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        gan_loss = self.gan_loss.forward_discriminator(d_real, d_fake)
        
        if use_multi_task:
            bio_loss_val = self.bio_loss(
                pred_qed_real, pred_sa_real, pred_logp_real,
                true_qed, true_sa, true_logp
            )
        else:
            bio_loss_val = torch.tensor(0.0, device=d_real.device)
        
        total_loss = self.lambda_gan * gan_loss + self.lambda_bio * bio_loss_val
        
        loss_dict = {
            'gan_loss': gan_loss.item(),
            'bio_loss': bio_loss_val.item(),
            'kg_loss': 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def generator_loss(
        self,
        d_fake: torch.Tensor,
        pred_qed_fake: torch.Tensor,
        pred_sa_fake: torch.Tensor,
        pred_logp_fake: torch.Tensor,
        h_mol: torch.Tensor,
        h_kg: torch.Tensor,
        valid_mask: List[int],
        batch_size: int,
        use_multi_task: bool = True,
        use_kg_consistency: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        gan_loss = self.gan_loss.forward_generator(d_fake)
        
        if use_multi_task:
            ideal_qed = [1.0] * batch_size
            ideal_sa = [1.0] * batch_size
            ideal_logp = [2.0] * batch_size
            
            bio_loss_val = self.bio_loss(
                pred_qed_fake, pred_sa_fake, pred_logp_fake,
                ideal_qed, ideal_sa, ideal_logp
            )
        else:
            bio_loss_val = torch.tensor(0.0, device=d_fake.device)
        
        chem_loss_val = self.chem_loss(valid_mask, batch_size)
        
        if use_kg_consistency:
            kg_loss_val = self.kg_loss(h_mol, h_kg)
        else:
            kg_loss_val = torch.tensor(0.0, device=d_fake.device)
        
        total_loss = (
            self.lambda_gan * gan_loss +
            self.alpha * bio_loss_val +
            self.lambda_chem * chem_loss_val +
            self.beta * kg_loss_val
        )
        
        loss_dict = {
            'gan_loss': gan_loss.item(),
            'bio_loss': bio_loss_val.item(),
            'chem_loss': chem_loss_val.item(),
            'kg_loss': kg_loss_val.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# ABLATION DATASET
# ============================================================================

class AblationDataset(torch.utils.data.Dataset):
    """Dataset wrapper for ablation study with KG support."""
    
    def __init__(self, data: Tuple, raw_smiles_list: List[str], 
                 protein_to_kg_mapping: Optional[Dict[str, str]] = None,
                 kg_dataset: Optional[KGDataset] = None):
        proIndices, smiIndices, labelIndices, proMask, smiMask = data
        self._len = len(proIndices)
        self.x = proIndices
        self.y = smiIndices
        self.label = labelIndices
        self.proMask = proMask
        self.smiMask = smiMask
        self.raw_smiles = raw_smiles_list
        self.protein_to_kg_mapping = protein_to_kg_mapping or {}
        self.kg_dataset = kg_dataset
        
        self.kg_entity_indices = []
        for i in range(self._len):
            protein = ''.join([kg_dataset.idx_to_entity.get(idx, '') for idx in proIndices[i] if idx != 0]) if kg_dataset else ''
            kg_entity_id = self.protein_to_kg_mapping.get(protein)
            
            if kg_entity_id and kg_dataset and kg_entity_id in kg_dataset.entity_to_idx:
                kg_idx = kg_dataset.entity_to_idx[kg_entity_id]
            else:
                kg_idx = -1
            
            self.kg_entity_indices.append(kg_idx)
    
    def __getitem__(self, idx: int):
        proMask = [1.0] * self.proMask[idx] + [0.0] * (len(self.x[idx]) - self.proMask[idx])
        smiMask = [1.0] * self.smiMask[idx] + [0.0] * (len(self.label[idx]) - self.smiMask[idx])
        
        return (
            self.x[idx],
            self.y[idx],
            self.label[idx],
            np.array(proMask).astype(int),
            np.array(smiMask).astype(int),
            self.raw_smiles[idx],
            self.kg_entity_indices[idx]
        )
    
    def __len__(self):
        return self._len


# ============================================================================
# KG UTILITIES
# ============================================================================

def pretrain_kg(kg_dataset: KGDataset, kg_dim: int = 100, epochs: int = 100,
                batch_size: int = 512, lr: float = 0.01, margin: float = 1.0,
                n_neg: int = 1, device: str = 'cpu',
                save_path: Optional[str] = None,
                verbose: bool = True) -> TransEModel:
    logger.info(f"Pre-training TransE: dim={kg_dim}, epochs={epochs}, lr={lr}, margin={margin}")
    
    transe_model = train_transe(
        kg_dataset=kg_dataset,
        embedding_dim=kg_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        n_neg=n_neg,
        device=device,
        save_path=save_path,
        verbose=verbose
    )
    
    logger.info(f"TransE pre-training completed. Model saved to {save_path}")
    return transe_model


def get_batch_kg_embeddings(transe_model: TransEModel, kg_dataset: KGDataset,
                            protein_entity_indices: torch.Tensor,
                            neighborhood_k: int = 1,
                            device: str = 'cpu') -> torch.Tensor:
    batch_size = len(protein_entity_indices)
    kg_embeddings = []
    
    for i in range(batch_size):
        idx = protein_entity_indices[i].item()
        entity_id = kg_dataset.idx_to_entity.get(idx)
        
        if entity_id is None or entity_id not in kg_dataset.entity_to_idx:
            kg_emb = torch.zeros(transe_model.embedding_dim, device=device)
        else:
            kg_emb = aggregate_neighborhood(
                transe_model,
                kg_dataset,
                entity_id,
                k=neighborhood_k
            ).to(device)
        
        kg_embeddings.append(kg_emb)
    
    return torch.stack(kg_embeddings, dim=0)


def prepare_kg_dataset(kg_triples_path: str) -> Optional[KGDataset]:
    if not os.path.exists(kg_triples_path):
        logger.warning(f"KG triples file not found: {kg_triples_path}")
        return None
    
    logger.info(f"Building KG from {kg_triples_path}")
    kg_dataset = build_kg_from_tsv(kg_triples_path)
    logger.info(f"KG built: {kg_dataset.num_entities} entities, {kg_dataset.num_relations} relations, {len(kg_dataset)} triples")
    
    return kg_dataset


def build_protein_to_kg_mapping(data_path: str, kg_dataset: KGDataset) -> Dict[str, str]:
    mapping = {}
    
    protein_entities = set()
    for entity_id, entity_type in kg_dataset.entity_types.items():
        if entity_type == 'Protein':
            protein_entities.add(entity_id)
    
    data = pd.read_csv(data_path, sep='\t')
    
    for idx, row in data.iterrows():
        protein_seq = row['protein']
        protein_key = protein_seq[:100] if len(protein_seq) > 100 else protein_seq
        
        for protein_entity in protein_entities:
            if protein_key in protein_entity or protein_entity in protein_key:
                mapping[protein_seq] = protein_entity
                break
        
        if protein_seq not in mapping:
            hash_id = f"PROT_{hash(protein_seq) % 1000000}"
            mapping[protein_seq] = hash_id
    
    logger.info(f"Built protein-to-KG mapping for {len(mapping)} proteins")
    return mapping


def prepare_ablation_dataset(config, kg_dataset: Optional[KGDataset],
                             protein_to_kg_mapping: Optional[Dict[str, str]],
                             orign: str = 'train',
                             data_dir: str = './data') -> Tuple:
    with open(os.path.join(data_dir, 'train-val-split.json'), 'r') as f:
        data_config = json.load(f)
    
    slices = data_config[orign]
    data = pd.read_csv(os.path.join(data_dir, 'train-val-data.tsv'), sep='\t')
    data = data.loc[slices]
    
    raw_smiles_list = data['smiles'].tolist()
    
    smiArr = data['smiles'].apply(splitSmi).tolist()
    proArr = data['protein'].apply(list).tolist()
    
    smiIndices, labelIndices, smiMask = fetchIndices(smiArr, config.smiVoc, config.smiMaxLen)
    proIndices, _, proMask = fetchIndices(proArr, config.proVoc, config.proMaxLen)
    
    return (proIndices, smiIndices, labelIndices, proMask, smiMask), raw_smiles_list


# ============================================================================
# TRAINING STEP FUNCTIONS
# ============================================================================

def train_discriminator_step(
    batch: Tuple,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    joint_loss: KGJointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    use_multi_task: bool = True,
    teacher_forcing: bool = True
) -> Dict[str, float]:
    (protein, smiles, label, pro_mask, smi_mask, raw_smiles, kg_entity_idx) = batch
    
    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    label = torch.as_tensor(label).to(device)
    
    batch_size = protein.size(0)
    
    protein_output = protein_encoder(protein, apply_mlm=False)
    h_seq = protein_output['protein_features']
    
    real_mol_output = smi_encoder(smiles)
    real_mol_features = real_mol_output['token_features']
    
    disc_real_output = discriminator(real_mol_features, h_seq)
    
    if teacher_forcing:
        tgt_mask = torch.triu(
            torch.ones(smiles.size(1), smiles.size(1), device=device),
            diagonal=1
        ).masked_fill(torch.ones(smiles.size(1), smiles.size(1), device=device) == 1, float('-inf'))
        
        gen_logits = generator(smiles, h_seq)
        
        if torch.rand(1).item() < 0.5:
            gen_probs = F.softmax(gen_logits, dim=-1)
            fake_smiles = torch.multinomial(gen_probs.view(-1, gen_probs.size(-1)), 1)
            fake_smiles = fake_smiles.view(batch_size, -1)
        else:
            fake_smiles = gen_logits.argmax(dim=-1)
    else:
        start_token = smiVoc.index('&')
        fake_smiles = generator.generate(h_seq, max_len=smiles.size(1), start_token=start_token)
    
    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    
    disc_fake_output = discriminator(fake_mol_features, h_seq)
    
    real_props = compute_mol_properties(raw_smiles)
    
    loss, loss_dict = joint_loss.discriminator_loss(
        d_real=disc_real_output['real_fake'],
        d_fake=disc_fake_output['real_fake'],
        pred_qed_real=disc_real_output['qed'],
        pred_sa_real=disc_real_output['sa'],
        pred_logp_real=disc_real_output['logp'],
        true_qed=real_props['qed'],
        true_sa=real_props['sa'],
        true_logp=real_props['logp'],
        use_multi_task=use_multi_task
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss_dict


def train_generator_step(
    batch: Tuple,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    joint_loss: KGJointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    use_multi_task: bool = True,
    use_kg_consistency: bool = False,
    h_kg: Optional[torch.Tensor] = None,
    teacher_forcing: bool = True
) -> Dict[str, float]:
    (protein, smiles, label, pro_mask, smi_mask, raw_smiles, kg_entity_idx) = batch
    
    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    
    batch_size = protein.size(0)
    
    protein_output = protein_encoder(protein, apply_mlm=False)
    h_seq = protein_output['protein_features']
    
    if teacher_forcing:
        gen_probs = F.softmax(generator(smiles, h_seq), dim=-1)
        fake_smiles = torch.multinomial(gen_probs.view(-1, gen_probs.size(-1)), 1)
        fake_smiles = fake_smiles.view(batch_size, -1)
    else:
        start_token = smiVoc.index('&')
        fake_smiles = generator.generate(h_seq, max_len=smiles.size(1), start_token=start_token)
    
    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    h_mol = fake_mol_output['molecular_features']
    
    disc_fake_output = discriminator(fake_mol_features, h_seq)
    
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
    for smi in fake_smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            valid_mask.append(1 if mol is not None else 0)
        except:
            valid_mask.append(0)
    
    h_kg_for_loss = h_kg if h_kg is not None else h_seq
    
    loss, loss_dict = joint_loss.generator_loss(
        d_fake=disc_fake_output['real_fake'],
        pred_qed_fake=disc_fake_output['qed'],
        pred_sa_fake=disc_fake_output['sa'],
        pred_logp_fake=disc_fake_output['logp'],
        h_mol=h_mol,
        h_kg=h_kg_for_loss,
        valid_mask=valid_mask,
        batch_size=batch_size,
        use_multi_task=use_multi_task,
        use_kg_consistency=use_kg_consistency
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    optimizer.step()
    
    validity_rate = sum(valid_mask) / max(batch_size, 1)
    loss_dict['validity_rate'] = validity_rate
    
    return loss_dict


def train_kg_discriminator_step(
    batch: Tuple,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    kg_condition_builder: Optional[KGConditionBuilder],
    joint_loss: KGJointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    teacher_forcing: bool = True
) -> Dict[str, float]:
    (protein, smiles, label, pro_mask, smi_mask, raw_smiles, kg_entity_idx) = batch
    
    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    label = torch.as_tensor(label).to(device)
    kg_entity_idx = torch.as_tensor(kg_entity_idx).to(device)
    
    batch_size = protein.size(0)
    
    protein_output = protein_encoder(protein, apply_mlm=False)
    h_seq = protein_output['protein_features']
    
    if kg_condition_builder is not None and (kg_entity_idx >= 0).any():
        kg_embeddings = get_batch_kg_embeddings(
            kg_condition_builder.transe_model,
            kg_condition_builder.kg_dataset,
            kg_entity_idx,
            neighborhood_k=kg_condition_builder.neighborhood_k,
            device=device
        )
        h_kg_projected = kg_condition_builder.projector(kg_embeddings)
        h_fuse, _ = kg_condition_builder.fusion(h_seq, h_kg_projected)
    else:
        h_fuse = h_seq
        h_kg_projected = None
    
    real_mol_output = smi_encoder(smiles)
    real_mol_features = real_mol_output['token_features']
    
    disc_real_output = discriminator(real_mol_features, h_fuse)
    
    if teacher_forcing:
        gen_logits = generator(smiles, h_fuse)
        
        if torch.rand(1).item() < 0.5:
            gen_probs = F.softmax(gen_logits, dim=-1)
            fake_smiles = torch.multinomial(gen_probs.view(-1, gen_probs.size(-1)), 1)
            fake_smiles = fake_smiles.view(batch_size, -1)
        else:
            fake_smiles = gen_logits.argmax(dim=-1)
    else:
        start_token = smiVoc.index('&')
        fake_smiles = generator.generate(h_fuse, max_len=smiles.size(1), start_token=start_token)
    
    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    
    disc_fake_output = discriminator(fake_mol_features, h_fuse)
    
    real_props = compute_mol_properties(raw_smiles)
    
    loss, loss_dict = joint_loss.discriminator_loss(
        d_real=disc_real_output['real_fake'],
        d_fake=disc_fake_output['real_fake'],
        pred_qed_real=disc_real_output['qed'],
        pred_sa_real=disc_real_output['sa'],
        pred_logp_real=disc_real_output['logp'],
        true_qed=real_props['qed'],
        true_sa=real_props['sa'],
        true_logp=real_props['logp']
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss_dict


def train_kg_generator_step(
    batch: Tuple,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    kg_condition_builder: Optional[KGConditionBuilder],
    joint_loss: KGJointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    use_kg_consistency: bool = True,
    teacher_forcing: bool = True
) -> Dict[str, float]:
    (protein, smiles, label, pro_mask, smi_mask, raw_smiles, kg_entity_idx) = batch
    
    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    kg_entity_idx = torch.as_tensor(kg_entity_idx).to(device)
    
    batch_size = protein.size(0)
    
    protein_output = protein_encoder(protein, apply_mlm=False)
    h_seq = protein_output['protein_features']
    
    if kg_condition_builder is not None and (kg_entity_idx >= 0).any():
        kg_embeddings = get_batch_kg_embeddings(
            kg_condition_builder.transe_model,
            kg_condition_builder.kg_dataset,
            kg_entity_idx,
            neighborhood_k=kg_condition_builder.neighborhood_k,
            device=device
        )
        h_kg_projected = kg_condition_builder.projector(kg_embeddings)
        h_fuse, _ = kg_condition_builder.fusion(h_seq, h_kg_projected)
    else:
        h_fuse = h_seq
        h_kg_projected = None
    
    if teacher_forcing:
        gen_probs = F.softmax(generator(smiles, h_fuse), dim=-1)
        fake_smiles = torch.multinomial(gen_probs.view(-1, gen_probs.size(-1)), 1)
        fake_smiles = fake_smiles.view(batch_size, -1)
    else:
        start_token = smiVoc.index('&')
        fake_smiles = generator.generate(h_fuse, max_len=smiles.size(1), start_token=start_token)
    
    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    h_mol = fake_mol_output['molecular_features']
    
    disc_fake_output = discriminator(fake_mol_features, h_fuse)
    
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
    for smi in fake_smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            valid_mask.append(1 if mol is not None else 0)
        except:
            valid_mask.append(0)
    
    h_kg_for_loss = h_kg_projected if h_kg_projected is not None else h_fuse
    
    loss, loss_dict = joint_loss.generator_loss(
        d_fake=disc_fake_output['real_fake'],
        pred_qed_fake=disc_fake_output['qed'],
        pred_sa_fake=disc_fake_output['sa'],
        pred_logp_fake=disc_fake_output['logp'],
        h_mol=h_mol,
        h_kg=h_kg_for_loss,
        valid_mask=valid_mask,
        batch_size=batch_size,
        use_kg_consistency=use_kg_consistency
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    optimizer.step()
    
    validity_rate = sum(valid_mask) / max(batch_size, 1)
    loss_dict['validity_rate'] = validity_rate
    
    return loss_dict


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

@torch.no_grad()
def validate_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    val_loader: DataLoader,
    joint_loss: KGJointLoss,
    device: torch.device,
    smiVoc: List[str],
    use_multi_task: bool = True,
    teacher_forcing: bool = True
) -> Dict[str, float]:
    generator.eval()
    discriminator.eval()
    protein_encoder.eval()
    smi_encoder.eval()
    
    val_losses = {
        'gen_total': 0.0,
        'gen_gan': 0.0,
        'gen_bio': 0.0,
        'gen_chem': 0.0,
        'gen_kg': 0.0,
        'disc_total': 0.0,
        'disc_gan': 0.0,
        'disc_bio': 0.0,
        'validity_rate': 0.0,
        'real_acc': 0.0,
        'fake_acc': 0.0,
        'qed': 0.0,
        'sa': 0.0,
        'logP': 0.0,
        'uniqueness': 0.0
    }
    
    num_batches = len(val_loader)
    all_fake_smiles = []
    all_qed = []
    all_sa = []
    all_logp = []
    
    for batch in tqdm(val_loader, desc='Validating'):
        (protein, smiles, label, pro_mask, smi_mask, raw_smiles, kg_entity_idx) = batch
        
        protein = protein.to(device)
        smiles = smiles.to(device)
        batch_size = protein.size(0)
        
        protein_output = protein_encoder(protein, apply_mlm=False)
        h_seq = protein_output['protein_features']
        
        if teacher_forcing:
            gen_logits = generator(smiles, h_seq)
            fake_smiles = gen_logits.argmax(dim=-1)
        else:
            start_token = smiVoc.index('&')
            fake_smiles = generator.generate(h_seq, max_len=smiles.size(1), start_token=start_token)
        
        real_mol_output = smi_encoder(smiles)
        real_mol_features = real_mol_output['token_features']
        
        fake_mol_output = smi_encoder(fake_smiles)
        fake_mol_features = fake_mol_output['token_features']
        h_mol = fake_mol_output['molecular_features']
        
        disc_real_output = discriminator(real_mol_features, h_seq)
        disc_fake_output = discriminator(fake_mol_features, h_seq)
        
        real_props = compute_mol_properties(raw_smiles)
        
        disc_loss, _ = joint_loss.discriminator_loss(
            d_real=disc_real_output['real_fake'],
            d_fake=disc_fake_output['real_fake'],
            pred_qed_real=disc_real_output['qed'],
            pred_sa_real=disc_real_output['sa'],
            pred_logp_real=disc_real_output['logp'],
            true_qed=real_props['qed'],
            true_sa=real_props['sa'],
            true_logp=real_props['logp'],
            use_multi_task=use_multi_task
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
        for smi in fake_smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                valid_mask.append(1 if mol is not None else 0)
            except:
                valid_mask.append(0)
        
        gen_loss, gen_loss_dict = joint_loss.generator_loss(
            d_fake=disc_fake_output['real_fake'],
            pred_qed_fake=disc_fake_output['qed'],
            pred_sa_fake=disc_fake_output['sa'],
            pred_logp_fake=disc_fake_output['logp'],
            h_mol=h_mol,
            h_kg=h_seq,
            valid_mask=valid_mask,
            batch_size=batch_size,
            use_multi_task=use_multi_task,
            use_kg_consistency=False
        )
        
        val_losses['disc_total'] += disc_loss.item()
        val_losses['gen_total'] += gen_loss.item()
        val_losses['gen_gan'] += gen_loss_dict['gan_loss']
        val_losses['gen_bio'] += gen_loss_dict['bio_loss']
        val_losses['gen_chem'] += gen_loss_dict['chem_loss']
        val_losses['gen_kg'] += gen_loss_dict['kg_loss']
        val_losses['validity_rate'] += sum(valid_mask) / max(batch_size, 1)
        
        real_acc = (disc_real_output['real_fake'] > 0.5).float().mean().item()
        fake_acc = (disc_fake_output['real_fake'] < 0.5).float().mean().item()
        val_losses['real_acc'] += real_acc
        val_losses['fake_acc'] += fake_acc
        
        all_fake_smiles.extend(fake_smiles_list)
        
        props = compute_mol_properties(fake_smiles_list)
        all_qed.extend(props['qed'])
        all_sa.extend(props['sa'])
        all_logp.extend(props['logp'])
    
    for key in val_losses:
        val_losses[key] /= num_batches
    
    if len(all_qed) > 0:
        val_losses['qed'] = np.mean(all_qed)
        val_losses['sa'] = np.mean(all_sa)
        val_losses['logP'] = np.mean(all_logp)
    
    if len(all_fake_smiles) > 0:
        val_losses['uniqueness'] = compute_uniqueness(all_fake_smiles)
    
    return val_losses


@torch.no_grad()
def validate_kg_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    kg_condition_builder: Optional[KGConditionBuilder],
    val_loader: DataLoader,
    joint_loss: KGJointLoss,
    device: torch.device,
    smiVoc: List[str],
    teacher_forcing: bool = True
) -> Dict[str, float]:
    generator.eval()
    discriminator.eval()
    protein_encoder.eval()
    smi_encoder.eval()
    
    val_losses = {
        'gen_total': 0.0,
        'gen_gan': 0.0,
        'gen_bio': 0.0,
        'gen_chem': 0.0,
        'gen_kg': 0.0,
        'disc_total': 0.0,
        'disc_gan': 0.0,
        'disc_bio': 0.0,
        'validity_rate': 0.0,
        'real_acc': 0.0,
        'fake_acc': 0.0,
        'qed': 0.0,
        'sa': 0.0,
        'logP': 0.0,
        'uniqueness': 0.0
    }
    
    num_batches = len(val_loader)
    all_fake_smiles = []
    all_qed = []
    all_sa = []
    all_logp = []
    
    for batch in tqdm(val_loader, desc='Validating'):
        (protein, smiles, label, pro_mask, smi_mask, raw_smiles, kg_entity_idx) = batch
        
        protein = protein.to(device)
        smiles = smiles.to(device)
        kg_entity_idx = kg_entity_idx.to(device)
        batch_size = protein.size(0)
        
        protein_output = protein_encoder(protein, apply_mlm=False)
        h_seq = protein_output['protein_features']
        
        if kg_condition_builder is not None and (kg_entity_idx >= 0).any():
            kg_embeddings = get_batch_kg_embeddings(
                kg_condition_builder.transe_model,
                kg_condition_builder.kg_dataset,
                kg_entity_idx,
                neighborhood_k=kg_condition_builder.neighborhood_k,
                device=device
            )
            h_kg_projected = kg_condition_builder.projector(kg_embeddings)
            h_fuse, _ = kg_condition_builder.fusion(h_seq, h_kg_projected)
        else:
            h_fuse = h_seq
            h_kg_projected = None
        
        if teacher_forcing:
            gen_logits = generator(smiles, h_fuse)
            fake_smiles = gen_logits.argmax(dim=-1)
        else:
            start_token = smiVoc.index('&')
            fake_smiles = generator.generate(h_fuse, max_len=smiles.size(1), start_token=start_token)
        
        real_mol_output = smi_encoder(smiles)
        real_mol_features = real_mol_output['token_features']
        
        fake_mol_output = smi_encoder(fake_smiles)
        fake_mol_features = fake_mol_output['token_features']
        h_mol = fake_mol_output['molecular_features']
        
        disc_real_output = discriminator(real_mol_features, h_fuse)
        disc_fake_output = discriminator(fake_mol_features, h_fuse)
        
        real_props = compute_mol_properties(raw_smiles)
        
        disc_loss, _ = joint_loss.discriminator_loss(
            d_real=disc_real_output['real_fake'],
            d_fake=disc_fake_output['real_fake'],
            pred_qed_real=disc_real_output['qed'],
            pred_sa_real=disc_real_output['sa'],
            pred_logp_real=disc_real_output['logp'],
            true_qed=real_props['qed'],
            true_sa=real_props['sa'],
            true_logp=real_props['logp']
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
        for smi in fake_smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                valid_mask.append(1 if mol is not None else 0)
            except:
                valid_mask.append(0)
        
        h_kg_for_loss = h_kg_projected if h_kg_projected is not None else h_fuse
        
        gen_loss, gen_loss_dict = joint_loss.generator_loss(
            d_fake=disc_fake_output['real_fake'],
            pred_qed_fake=disc_fake_output['qed'],
            pred_sa_fake=disc_fake_output['sa'],
            pred_logp_fake=disc_fake_output['logp'],
            h_mol=h_mol,
            h_kg=h_kg_for_loss,
            valid_mask=valid_mask,
            batch_size=batch_size,
            use_kg_consistency=True
        )
        
        val_losses['disc_total'] += disc_loss.item()
        val_losses['gen_total'] += gen_loss.item()
        val_losses['gen_gan'] += gen_loss_dict['gan_loss']
        val_losses['gen_bio'] += gen_loss_dict['bio_loss']
        val_losses['gen_chem'] += gen_loss_dict['chem_loss']
        val_losses['gen_kg'] += gen_loss_dict['kg_loss']
        val_losses['validity_rate'] += sum(valid_mask) / max(batch_size, 1)
        
        real_acc = (disc_real_output['real_fake'] > 0.5).float().mean().item()
        fake_acc = (disc_fake_output['real_fake'] < 0.5).float().mean().item()
        val_losses['real_acc'] += real_acc
        val_losses['fake_acc'] += fake_acc
        
        all_fake_smiles.extend(fake_smiles_list)
        
        props = compute_mol_properties(fake_smiles_list)
        all_qed.extend(props['qed'])
        all_sa.extend(props['sa'])
        all_logp.extend(props['logp'])
    
    for key in val_losses:
        val_losses[key] /= num_batches
    
    if len(all_qed) > 0:
        val_losses['qed'] = np.mean(all_qed)
        val_losses['sa'] = np.mean(all_sa)
        val_losses['logP'] = np.mean(all_logp)
    
    if len(all_fake_smiles) > 0:
        val_losses['uniqueness'] = compute_uniqueness(all_fake_smiles)
    
    return val_losses


# ============================================================================
# CHAPTER 3 ABLATION - DrugGAN-MSM
# ============================================================================

def run_ch3_ablation(
    config_name: str,
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
    use_mlm: bool = True,
    use_sa: bool = True,
    use_multi_task: bool = True,
    use_joint_loss: bool = True
) -> Dict[str, Any]:
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Chapter 3 Ablation: {config_name}")
    logger.info(f"  MLM: {use_mlm}, SA: {use_sa}, Multi-task D: {use_multi_task}, Joint Loss: {use_joint_loss}")
    logger.info(f"{'='*60}")
    
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(os.path.join(config_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(config_dir, 'logs'), exist_ok=True)
    
    logger.add(os.path.join(config_dir, 'logs', 'train.log'))
    
    # Select protein encoder variant based on ablation config
    if use_mlm and use_sa:
        protein_encoder = ProteinEncoder(
            pro_voc_len=len(config.proVoc),
            d_model=args.d_model,
            mask_rate=args.mask_rate
        )
    elif not use_mlm:
        protein_encoder = ProteinEncoderNoMLM(
            pro_voc_len=len(config.proVoc),
            d_model=args.d_model
        )
    else:
        protein_encoder = ProteinEncoderNoSA(
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
    
    if use_multi_task:
        discriminator = MultiTaskDiscriminator(
            d_model=args.d_model,
            hidden_dim=args.d_model // 2
        )
    else:
        discriminator = SimpleDiscriminator(
            protein_dim=756,
            hidden_dim=args.d_model // 2
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
    
    lambda_bio = 0.3 if use_joint_loss else 0.0
    lambda_chem = 0.2 if use_joint_loss else 0.0
    
    joint_loss = KGJointLoss(
        lambda_gan=1.0,
        alpha=0.5,
        beta=0.0,
        lambda_bio=lambda_bio,
        lambda_chem=lambda_chem
    )
    
    train_history = []
    val_history = []
    results = []
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        protein_encoder.train()
        smi_encoder.train()
        generator.train()
        discriminator.train()
        
        epoch_losses = {
            'gen_total': 0.0, 'gen_gan': 0.0, 'gen_bio': 0.0, 'gen_chem': 0.0, 'gen_kg': 0.0,
            'disc_total': 0.0, 'disc_gan': 0.0, 'disc_bio': 0.0, 'validity_rate': 0.0
        }
        num_batches = len(train_loader)
        
        for batch in tqdm(train_loader, desc='Training'):
            batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
            
            for _ in range(args.n_critic):
                disc_losses = train_discriminator_step(
                    batch, protein_encoder, smi_encoder, generator, discriminator,
                    joint_loss, disc_optimizer, device, config.smiVoc,
                    use_multi_task=use_multi_task, teacher_forcing=args.teacher_forcing
                )
                epoch_losses['disc_total'] += disc_losses['total_loss']
                epoch_losses['disc_gan'] += disc_losses['gan_loss']
                epoch_losses['disc_bio'] += disc_losses['bio_loss']
            
            gen_losses = train_generator_step(
                batch, protein_encoder, smi_encoder, generator, discriminator,
                joint_loss, gen_optimizer, device, config.smiVoc,
                use_multi_task=use_multi_task, use_kg_consistency=False,
                teacher_forcing=args.teacher_forcing
            )
            epoch_losses['gen_total'] += gen_losses['total_loss']
            epoch_losses['gen_gan'] += gen_losses['gan_loss']
            epoch_losses['gen_bio'] += gen_losses['bio_loss']
            epoch_losses['gen_chem'] += gen_losses['chem_loss']
            epoch_losses['gen_kg'] += gen_losses['kg_loss']
            epoch_losses['validity_rate'] += gen_losses.get('validity_rate', 0.0)
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        val_losses = validate_epoch(
            generator, discriminator, protein_encoder, smi_encoder,
            val_loader, joint_loss, device, config.smiVoc,
            use_multi_task=use_multi_task, teacher_forcing=args.teacher_forcing
        )
        
        train_history.append(epoch_losses)
        val_history.append(val_losses)
        
        result_row = {
            'config_name': config_name,
            'epoch': epoch + 1,
            'val_loss': val_losses['gen_total'],
            'real_acc': val_losses['real_acc'],
            'fake_acc': val_losses['fake_acc'],
            'qed': val_losses['qed'],
            'sa': val_losses['sa'],
            'logP': val_losses['logP'],
            'validity': val_losses['validity_rate'],
            'uniqueness': val_losses['uniqueness'],
            'kg_loss': val_losses['gen_kg']
        }
        results.append(result_row)
        
        logger.info(f"Train Gen: {epoch_losses['gen_total']:.4f}, Disc: {epoch_losses['disc_total']:.4f}")
        logger.info(f"Val Gen: {val_losses['gen_total']:.4f}, QED: {val_losses['qed']:.4f}, SA: {val_losses['sa']:.4f}, LogP: {val_losses['logP']:.4f}")
        
        gen_scheduler.step()
        disc_scheduler.step()
        
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'protein_encoder_state_dict': protein_encoder.state_dict(),
                'smi_encoder_state_dict': smi_encoder.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'config': {'use_mlm': use_mlm, 'use_sa': use_sa, 'use_multi_task': use_multi_task, 'use_joint_loss': use_joint_loss}
            }
            torch.save(checkpoint, os.path.join(config_dir, 'model', f'epoch_{epoch+1}.pt'))
    
    best_val = min(val_history, key=lambda x: x['gen_total'])
    logger.info(f"\nBest validation loss: {best_val['gen_total']:.4f} at epoch {val_history.index(best_val)+1}")
    
    return {
        'config_name': config_name,
        'train_history': train_history,
        'val_history': val_history,
        'results': results,
        'best_val_loss': best_val['gen_total'],
        'best_epoch': val_history.index(best_val) + 1
    }


# ============================================================================
# CHAPTER 4 ABLATION - KG-DrugGAN-MSM
# ============================================================================

def run_ch4_ablation(
    config_name: str,
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    kg_dataset: Optional[KGDataset],
    protein_to_kg_mapping: Optional[Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
    use_kg_fusion: bool = True,
    use_kg_consistency: bool = True,
    use_pretrained_transe: bool = True
) -> Dict[str, Any]:
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Chapter 4 Ablation: {config_name}")
    logger.info(f"  KG Fusion: {use_kg_fusion}, KG Consistency: {use_kg_consistency}, Pretrained TransE: {use_pretrained_transe}")
    logger.info(f"{'='*60}")
    
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(os.path.join(config_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(config_dir, 'logs'), exist_ok=True)
    
    logger.add(os.path.join(config_dir, 'logs', 'train.log'))
    
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
        hidden_dim=args.d_model // 2
    )
    
    kg_condition_builder = None
    transe_model = None
    
    if use_kg_fusion and kg_dataset is not None:
        if use_pretrained_transe and os.path.exists(args.kg_triples_path):
            transe_save_path = os.path.join(config_dir, 'model', 'transe_best.pt')
            transe_model = pretrain_kg(
                kg_dataset=kg_dataset,
                kg_dim=args.kg_embedding_dim,
                epochs=args.kg_epochs,
                batch_size=512,
                lr=args.kg_lr,
                margin=args.kg_margin,
                device=device,
                save_path=transe_save_path,
                verbose=True
            )
        else:
            transe_model = TransEModel(
                num_entities=kg_dataset.num_entities,
                num_relations=kg_dataset.num_relations,
                embedding_dim=args.kg_embedding_dim
            )
            transe_model.to(device)
        
        kg_adjacency = {
            kg_dataset.entity_to_idx[e]: [kg_dataset.entity_to_idx[n] for n in neighbors]
            for e, rels in kg_dataset.adjacency.items() for neighbors in rels.values()
        }
        
        kg_condition_builder = KGConditionBuilder(
            entity_embeddings=transe_model.get_all_entity_embeddings(),
            kg_adjacency=kg_adjacency,
            d_model=args.d_model,
            kg_dim=args.kg_embedding_dim
        ).to(device)
    
    protein_encoder = protein_encoder.to(device)
    smi_encoder = smi_encoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    gen_params = list(generator.parameters())
    disc_params = list(discriminator.parameters()) + list(protein_encoder.parameters()) + list(smi_encoder.parameters())
    
    if kg_condition_builder is not None:
        gen_params += list(kg_condition_builder.parameters())
    
    gen_optimizer = optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_params, lr=args.lr, betas=(0.5, 0.999))
    
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=args.epochs, eta_min=1e-6)
    disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=args.epochs, eta_min=1e-6)
    
    joint_loss = KGJointLoss(
        lambda_gan=1.0,
        alpha=0.5,
        beta=0.5 if use_kg_consistency else 0.0,
        lambda_bio=0.3,
        lambda_chem=0.2
    )
    
    train_history = []
    val_history = []
    results = []
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        protein_encoder.train()
        smi_encoder.train()
        generator.train()
        discriminator.train()
        
        epoch_losses = {
            'gen_total': 0.0, 'gen_gan': 0.0, 'gen_bio': 0.0, 'gen_chem': 0.0, 'gen_kg': 0.0,
            'disc_total': 0.0, 'disc_gan': 0.0, 'disc_bio': 0.0, 'validity_rate': 0.0
        }
        num_batches = len(train_loader)
        
        for batch in tqdm(train_loader, desc='Training'):
            batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
            
            for _ in range(args.n_critic):
                disc_losses = train_kg_discriminator_step(
                    batch, protein_encoder, smi_encoder, generator, discriminator,
                    kg_condition_builder, joint_loss, disc_optimizer, device, config.smiVoc,
                    teacher_forcing=args.teacher_forcing
                )
                epoch_losses['disc_total'] += disc_losses['total_loss']
                epoch_losses['disc_gan'] += disc_losses['gan_loss']
                epoch_losses['disc_bio'] += disc_losses['bio_loss']
            
            gen_losses = train_kg_generator_step(
                batch, protein_encoder, smi_encoder, generator, discriminator,
                kg_condition_builder, joint_loss, gen_optimizer, device, config.smiVoc,
                use_kg_consistency=use_kg_consistency, teacher_forcing=args.teacher_forcing
            )
            epoch_losses['gen_total'] += gen_losses['total_loss']
            epoch_losses['gen_gan'] += gen_losses['gan_loss']
            epoch_losses['gen_bio'] += gen_losses['bio_loss']
            epoch_losses['gen_chem'] += gen_losses['chem_loss']
            epoch_losses['gen_kg'] += gen_losses['kg_loss']
            epoch_losses['validity_rate'] += gen_losses.get('validity_rate', 0.0)
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        val_losses = validate_kg_epoch(
            generator, discriminator, protein_encoder, smi_encoder,
            kg_condition_builder, val_loader, joint_loss, device, config.smiVoc,
            teacher_forcing=args.teacher_forcing
        )
        
        train_history.append(epoch_losses)
        val_history.append(val_losses)
        
        result_row = {
            'config_name': config_name,
            'epoch': epoch + 1,
            'val_loss': val_losses['gen_total'],
            'real_acc': val_losses['real_acc'],
            'fake_acc': val_losses['fake_acc'],
            'qed': val_losses['qed'],
            'sa': val_losses['sa'],
            'logP': val_losses['logP'],
            'validity': val_losses['validity_rate'],
            'uniqueness': val_losses['uniqueness'],
            'kg_loss': val_losses['gen_kg']
        }
        results.append(result_row)
        
        logger.info(f"Train Gen: {epoch_losses['gen_total']:.4f}, Disc: {epoch_losses['disc_total']:.4f}, KG: {epoch_losses['gen_kg']:.4f}")
        logger.info(f"Val Gen: {val_losses['gen_total']:.4f}, QED: {val_losses['qed']:.4f}, KG: {val_losses['gen_kg']:.4f}")
        
        gen_scheduler.step()
        disc_scheduler.step()
        
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'protein_encoder_state_dict': protein_encoder.state_dict(),
                'smi_encoder_state_dict': smi_encoder.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'config': {'use_kg_fusion': use_kg_fusion, 'use_kg_consistency': use_kg_consistency, 'use_pretrained_transe': use_pretrained_transe}
            }
            if kg_condition_builder is not None:
                checkpoint['kg_projector_state_dict'] = kg_condition_builder.projector.state_dict()
                checkpoint['gated_fusion_state_dict'] = kg_condition_builder.fusion.state_dict()
            if transe_model is not None:
                checkpoint['transe_state_dict'] = transe_model.state_dict()
            torch.save(checkpoint, os.path.join(config_dir, 'model', f'epoch_{epoch+1}.pt'))
    
    best_val = min(val_history, key=lambda x: x['gen_total'])
    logger.info(f"\nBest validation loss: {best_val['gen_total']:.4f} at epoch {val_history.index(best_val)+1}")
    
    return {
        'config_name': config_name,
        'train_history': train_history,
        'val_history': val_history,
        'results': results,
        'best_val_loss': best_val['gen_total'],
        'best_epoch': val_history.index(best_val) + 1
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation Study for DrugGAN-MSM and KG-DrugGAN-MSM')
    
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--ablation-mode', type=str, default='all', choices=['ch3', 'ch4', 'all'],
                        help='Which ablation study to run: ch3 (Chapter 3), ch4 (Chapter 4), or all')
    parser.add_argument('--output-dir', type=str, default='./ablation_output',
                        help='Output directory for results')
    parser.add_argument('--small-epochs', type=int, default=5,
                        help='Number of epochs for each ablation config (quick mode)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (if specified, overrides --small-epochs)')
    
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-critic', type=int, default=1, help='Number of critic updates per generator update')
    
    parser.add_argument('--d-model', type=int, default=512, help='Model hidden dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of protein encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=12, help='Number of generator decoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--noise-dim', type=int, default=128, help='Noise vector dimension')
    parser.add_argument('--mask-rate', type=float, default=0.15, help='MLM mask rate')
    
    parser.add_argument('--kg-embedding-dim', type=int, default=100, help='TransE embedding dimension')
    parser.add_argument('--kg-epochs', type=int, default=50, help='TransE pre-training epochs')
    parser.add_argument('--kg-lr', type=float, default=0.01, help='TransE learning rate')
    parser.add_argument('--kg-margin', type=float, default=1.0, help='TransE margin')
    parser.add_argument('--kg-triples-path', type=str, default='./kg/kg_triples.csv', help='Path to KG triples CSV file')
    parser.add_argument('--kg-neighborhood-k', type=int, default=1, help='k-hop neighborhood size')
    
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--teacher-forcing', action='store_true', default=True, help='Use teacher forcing')
    
    args = parser.parse_args()
    
    if args.epochs is not None:
        args.epochs = args.epochs
    else:
        args.epochs = args.small_epochs
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.add(os.path.join(args.output_dir, 'ablation.log'))
    logger.info(f"Ablation study started at {datetime.now()}")
    logger.info(f"Arguments: {args}")
    
    config = loadConfig(args)
    config.batchSize = args.batch_size
    
    kg_dataset = None
    protein_to_kg_mapping = None
    
    if args.ablation_mode in ['ch4', 'all']:
        if os.path.exists(args.kg_triples_path):
            kg_dataset = prepare_kg_dataset(args.kg_triples_path)
            protein_to_kg_mapping = build_protein_to_kg_mapping(os.path.join(args.data_dir, 'train-val-data.tsv'), kg_dataset)
        else:
            logger.warning(f"KG triples not found at {args.kg_triples_path}. Chapter 4 ablations will use random embeddings.")
    
    logger.info("Preparing datasets...")
    train_data, train_smiles = prepare_ablation_dataset(config, kg_dataset, protein_to_kg_mapping, 'train', data_dir=args.data_dir)
    val_data, val_smiles = prepare_ablation_dataset(config, kg_dataset, protein_to_kg_mapping, 'valid', data_dir=args.data_dir)
    
    train_dataset = AblationDataset(train_data, train_smiles, protein_to_kg_mapping, kg_dataset)
    val_dataset = AblationDataset(val_data, val_smiles, protein_to_kg_mapping, kg_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    all_results = []
    summary = []
    
    if args.ablation_mode in ['ch3', 'all']:
        logger.info("\n" + "="*80)
        logger.info("CHAPTER 3 ABLATION STUDIES (DrugGAN-MSM)")
        logger.info("="*80)
        
        ch3_configs = [
            ('ch3_full_model', True, True, True, True),
            ('ch3_w/o_MLM', False, True, True, True),
            ('ch3_w/o_SA', True, False, True, True),
            ('ch3_w/o_Multi-task_D', True, True, False, True),
            ('ch3_w/o_Joint_Loss', True, True, True, False)
        ]
        
        for config_name, use_mlm, use_sa, use_multi_task, use_joint_loss in ch3_configs:
            result = run_ch3_ablation(
                config_name, config, train_loader, val_loader, args, device, args.output_dir,
                use_mlm=use_mlm, use_sa=use_sa, use_multi_task=use_multi_task, use_joint_loss=use_joint_loss
            )
            all_results.extend(result['results'])
            summary.append({
                'config': config_name,
                'best_val_loss': result['best_val_loss'],
                'best_epoch': result['best_epoch'],
                'final_qed': result['val_history'][-1]['qed'],
                'final_sa': result['val_history'][-1]['sa'],
                'final_logP': result['val_history'][-1]['logP'],
                'final_validity': result['val_history'][-1]['validity_rate'],
                'final_uniqueness': result['val_history'][-1]['uniqueness']
            })
    
    if args.ablation_mode in ['ch4', 'all']:
        logger.info("\n" + "="*80)
        logger.info("CHAPTER 4 ABLATION STUDIES (KG-DrugGAN-MSM)")
        logger.info("="*80)
        
        ch4_configs = [
            ('ch4_full_KG_model', True, True, True),
            ('ch4_w/o_KG_Fusion', False, False, True),
            ('ch4_w/o_KG_Consistency', True, False, True),
            ('ch4_w/o_TransE_Pretraining', True, True, False)
        ]
        
        for config_name, use_kg_fusion, use_kg_consistency, use_pretrained_transe in ch4_configs:
            result = run_ch4_ablation(
                config_name, config, train_loader, val_loader, kg_dataset, protein_to_kg_mapping,
                args, device, args.output_dir,
                use_kg_fusion=use_kg_fusion, use_kg_consistency=use_kg_consistency, use_pretrained_transe=use_pretrained_transe
            )
            all_results.extend(result['results'])
            summary.append({
                'config': config_name,
                'best_val_loss': result['best_val_loss'],
                'best_epoch': result['best_epoch'],
                'final_qed': result['val_history'][-1]['qed'],
                'final_sa': result['val_history'][-1]['sa'],
                'final_logP': result['val_history'][-1]['logP'],
                'final_validity': result['val_history'][-1]['validity_rate'],
                'final_uniqueness': result['val_history'][-1]['uniqueness'],
                'final_kg_loss': result['val_history'][-1]['gen_kg']
            })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.output_dir, 'ablation_results.csv'), index=False)
    logger.info(f"\nSaved detailed results to {os.path.join(args.output_dir, 'ablation_results.csv')}")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(args.output_dir, 'ablation_summary.csv'), index=False)
    logger.info(f"Saved summary to {os.path.join(args.output_dir, 'ablation_summary.csv')}")
    
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("BEST METRICS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Best Val Loss':<15} {'QED':<8} {'SA':<8} {'LogP':<8} {'Validity':<10} {'Uniqueness':<12}")
    print("-"*100)
    
    for row in summary:
        kg_loss_str = f" {row.get('final_kg_loss', 0):.4f}" if 'kg_loss' in row.get('config', '') else ""
        print(f"{row['config']:<30} {row['best_val_loss']:<15.4f} {row['final_qed']:<8.4f} {row['final_sa']:<8.4f} {row['final_logP']:<8.4f} {row['final_validity']:<10.4f} {row['final_uniqueness']:<12.4f}{kg_loss_str}")
    
    print("="*80)
    
    logger.info(f"\nAblation study completed at {datetime.now()}")


if __name__ == '__main__':
    main()