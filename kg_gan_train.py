"""
KG-Enhanced GAN Training Script for DrugGAN-MSM

This script implements the training procedure for the KG-DrugGAN-MSM model
as described in Chapter 4 of the thesis. It trains the Generator and 
MultiTaskDiscriminator with knowledge graph enhancement using the joint loss:

    L_total = L_adv + α·L_prop + β·L_kg

Where:
- L_adv: Adversarial GAN loss
- L_prop: Biology property loss (QED, SA, logP)
- L_kg: Knowledge consistency loss (||h_mol - h̃_kg||²)
- α, β: Loss balancing weights (default: α=0.5, β=0.5)

The training pipeline includes:
1. Pre-training TransE on KG triples
2. Building knowledge representations for proteins
3. Training KG-enhanced GAN with gated fusion
4. Knowledge consistency constraint enforcement

Author: DrugGAN-MSM Team
"""

import os
import sys
import json
import time
import argparse
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
from rdkit.Chem import QED, Descriptors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.KG_DrugGAN_MSM import (
    KGDrugGANModel, 
    GatedFusion, 
    KnowledgeProjector, 
    KnowledgeConsistencyLoss,
    KGConditionBuilder
)
from kg.kg_dataset import KGDataset, build_kg_from_tsv
from kg.kg_transe import TransEModel, train_transe, aggregate_neighborhood, KnowledgeProjector as KGProjector
from utils.baseline import prepareDataset, loadConfig, MyDataset, fetchIndices, splitSmi
from utils.sascorer import calculateScore as compute_sa_score

from model.train_utils import GANLoss, BiologyPropertyLoss, ChemicalValidityLoss, compute_mol_properties, set_seed


def pretrain_kg(kg_dataset: KGDataset, kg_dim: int = 100, epochs: int = 100,
                batch_size: int = 512, lr: float = 0.01, margin: float = 1.0,
                n_neg: int = 1, device: str = 'cpu', 
                save_path: Optional[str] = None,
                verbose: bool = True) -> TransEModel:
    """
    Pre-train TransE on the knowledge graph.
    
    Args:
        kg_dataset: KG dataset with triples
        kg_dim: TransE embedding dimension (default: 100)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 512)
        lr: Learning rate (default: 0.01)
        margin: TransE margin (default: 1.0)
        n_neg: Number of negative samples per positive (default: 1)
        device: Device to train on
        save_path: Path to save trained model
        verbose: Print training progress
        
    Returns:
        Trained TransEModel
    """
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
    """
    For a batch of protein entity indices:
    1. Get entity embeddings directly from TransE model (vectorized)
    2. For entities not in KG, return zero embeddings
    3. Optionally aggregate k-hop neighborhood (still uses loop for graph traversal)
    
    Args:
        transe_model: Trained TransE model
        kg_dataset: KG dataset with adjacency information
        protein_entity_indices: Protein entity indices in KG vocab
        neighborhood_k: Number of hops for neighborhood aggregation
        device: Device for computation
        
    Returns:
        KG embeddings tensor (batch, kg_dim)
    """
    batch_size = len(protein_entity_indices)
    kg_dim = transe_model.embedding_dim
    
    # Get all entity embeddings at once (vectorized lookup)
    all_entity_embs = transe_model.get_all_entity_embeddings().to(device)
    
    # Create output tensor
    kg_embeddings = torch.zeros(batch_size, kg_dim, device=device)
    
    # Find valid entities (those in KG)
    valid_mask = protein_entity_indices >= 0
    valid_indices = protein_entity_indices[valid_mask]
    
    if valid_indices.numel() > 0:
        # Direct entity embeddings (vectorized)
        if neighborhood_k == 0:
            # Just use entity embeddings directly
            kg_embeddings[valid_mask] = all_entity_embs[valid_indices]
        else:
            # For neighborhood aggregation, still need to loop (graph traversal)
            # but we can use the vectorized entity embeddings for neighbors
            for i in range(batch_size):
                idx = protein_entity_indices[i].item()
                
                if idx < 0:
                    continue
                
                entity_id = kg_dataset.idx_to_entity.get(idx)
                
                if entity_id is None or entity_id not in kg_dataset.entity_to_idx:
                    continue
                
                # Use aggregate_neighborhood for k-hop aggregation
                kg_emb = aggregate_neighborhood(
                    transe_model,
                    kg_dataset,
                    entity_id,
                    k=neighborhood_k
                ).to(device)
                
                kg_embeddings[i] = kg_emb
    
    return kg_embeddings  # (batch, kg_dim)


class KGGANDataset(torch.utils.data.Dataset):
    """
    Dataset for KG-enhanced GAN training that includes protein entity indices.
    
    Extends the baseline dataset to provide access to:
    - Raw SMILES strings for property computation
    - Protein entity indices for KG lookup
    """
    
    def __init__(self, data: Tuple, raw_smiles_list: List[str], 
                 protein_to_kg_mapping: Dict[str, str],
                 kg_dataset: Optional[KGDataset] = None):
        """
        Initialize dataset.
        
        Args:
            data: Tuple of (proIndices, smiIndices, labelIndices, proMask, smiMask)
            raw_smiles_list: List of raw SMILES strings for property computation
            protein_to_kg_mapping: Mapping from protein sequence to KG entity ID
            kg_dataset: Optional KG dataset for entity index lookup
        """
        proIndices, smiIndices, labelIndices, proMask, smiMask = data
        self._len = len(proIndices)
        self.x = proIndices
        self.y = smiIndices
        self.label = labelIndices
        self.proMask = proMask
        self.smiMask = smiMask
        self.raw_smiles = raw_smiles_list
        self.protein_to_kg_mapping = protein_to_kg_mapping
        self.kg_dataset = kg_dataset
        
        # Pre-compute KG entity indices for proteins
        self.kg_entity_indices = []
        for i in range(self._len):
            protein = ''.join([kg_dataset.idx_to_entity.get(idx, '') for idx in proIndices[i] if idx != 0])
            kg_entity_id = protein_to_kg_mapping.get(protein)
            
            if kg_entity_id and kg_dataset and kg_entity_id in kg_dataset.entity_to_idx:
                kg_idx = kg_dataset.entity_to_idx[kg_entity_id]
            else:
                kg_idx = -1  # Invalid index for missing entities
            
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


def prepare_kg_dataset(kg_triples_path: str) -> KGDataset:
    """
    Prepare KG dataset from TSV file.
    
    Args:
        kg_triples_path: Path to KG triples TSV file
        
    Returns:
        KGDataset
    """
    if not os.path.exists(kg_triples_path):
        logger.warning(f"KG triples file not found: {kg_triples_path}")
        return None
    
    logger.info(f"Building KG from {kg_triples_path}")
    kg_dataset = build_kg_from_tsv(kg_triples_path)
    logger.info(f"KG built: {kg_dataset.num_entities} entities, {kg_dataset.num_relations} relations, {len(kg_dataset)} triples")
    
    return kg_dataset


def build_protein_to_kg_mapping(data_path: str, kg_dataset: KGDataset) -> Dict[str, str]:
    """
    Build mapping from protein sequence to KG entity ID.
    
    Simple heuristic:
    - Use protein sequence hash as a fallback if no direct mapping
    - Try to match protein sequences with UniProt IDs in KG
    
    Args:
        data_path: Path to train-val-data.tsv
        kg_dataset: KG dataset
        
    Returns:
        Dictionary mapping protein sequence to KG entity ID
    """
    mapping = {}
    
    # Build reverse mapping from entity type to entities
    protein_entities = set()
    for entity_id, entity_type in kg_dataset.entity_types.items():
        if entity_type == 'Protein':
            protein_entities.add(entity_id)
    
    # Load data and try to match
    data = pd.read_csv(data_path, sep='\t')
    
    for idx, row in data.iterrows():
        protein_seq = row['protein']
        
        # Simple heuristic: use first 100 chars as identifier
        protein_key = protein_seq[:100] if len(protein_seq) > 100 else protein_seq
        
        # Try to find matching protein in KG
        # In practice, you'd use UniProt ID mapping
        for protein_entity in protein_entities:
            if protein_key in protein_entity or protein_entity in protein_key:
                mapping[protein_seq] = protein_entity
                break
        
        # Fallback: use hash-based identifier
        if protein_seq not in mapping:
            # Create a pseudo-entity ID based on sequence hash
            hash_id = f"PROT_{hash(protein_seq) % 1000000}"
            mapping[protein_seq] = hash_id
    
    logger.info(f"Built protein-to-KG mapping for {len(mapping)} proteins")
    return mapping


def prepareKGGANDataset(config, kg_dataset: KGDataset, 
                        protein_to_kg_mapping: Dict[str, str],
                        orign: str = 'train',
                        data_dir: str = './data') -> Tuple:
    """
    Prepare dataset for KG-enhanced GAN training.
    
    Args:
        config: Configuration object with vocabularies and max lengths
        kg_dataset: KG dataset
        protein_to_kg_mapping: Mapping from protein to KG entity ID
        orign: Dataset split name ('train' or 'valid')
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (encoded_data, raw_smiles_list)
    """
    import json
    
    with open(os.path.join(data_dir, 'train-val-split.json'), 'r') as f:
        data_config = json.load(f)
    
    slices = data_config[orign]
    data = pd.read_csv(os.path.join(data_dir, 'train-val-data.tsv'), sep='\t')
    data = data.loc[slices]
    
    # Get raw SMILES
    raw_smiles_list = data['smiles'].tolist()
    
    # Process sequences
    smiArr = data['smiles'].apply(splitSmi).tolist()
    proArr = data['protein'].apply(list).tolist()
    
    # Encode sequences
    smiIndices, labelIndices, smiMask = fetchIndices(smiArr, config.smiVoc, config.smiMaxLen)
    proIndices, _, proMask = fetchIndices(proArr, config.proVoc, config.proMaxLen)
    
    return (proIndices, smiIndices, labelIndices, proMask, smiMask), raw_smiles_list




class KnowledgeConsistencyLossWrapper(nn.Module):
    """
    Wrapper for knowledge consistency loss.
    
    L_kg = ||h_mol - h̃_kg||²
    """
    
    def __init__(self):
        super(KnowledgeConsistencyLossWrapper, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, h_mol: torch.Tensor, h_kg: torch.Tensor) -> torch.Tensor:
        if h_mol.dim() > 2:
            h_mol = h_mol.mean(dim=1)
        return self.mse_loss(h_mol, h_kg)


class KGJointLoss(nn.Module):
    """
    Joint loss for KG-DrugGAN-MSM.
    
    L_total = L_adv + α·L_prop + β·L_kg
    """
    
    def __init__(self, lambda_gan: float = 1.0, alpha: float = 0.5, beta: float = 0.5,
                 lambda_bio: float = 0.5, lambda_chem: float = 0.5):
        super(KGJointLoss, self).__init__()
        self.lambda_gan = lambda_gan
        self.alpha = alpha  # Weight for property loss
        self.beta = beta    # Weight for knowledge consistency loss
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
        pred_affinity_real: torch.Tensor,
        true_qed: List[float],
        true_sa: List[float],
        true_logp: List[float],
        true_affinity: List[float]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        gan_loss = self.gan_loss.forward_discriminator(d_real, d_fake)
        bio_loss_val = self.bio_loss(
            pred_qed_real, pred_sa_real, pred_logp_real, pred_affinity_real,
            true_qed, true_sa, true_logp, true_affinity
        )
        
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
        pred_affinity_fake: torch.Tensor,
        h_mol: torch.Tensor,
        h_kg: torch.Tensor,
        valid_mask: List[int],
        batch_size: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        gan_loss = self.gan_loss.forward_generator(d_fake)
        
        ideal_qed = [1.0] * batch_size
        ideal_sa = [1.0] * batch_size
        ideal_logp = [2.0] * batch_size
        ideal_affinity = [-10.0] * batch_size
        
        bio_loss_val = self.bio_loss(
            pred_qed_fake, pred_sa_fake, pred_logp_fake, pred_affinity_fake,
            ideal_qed, ideal_sa, ideal_logp, ideal_affinity
        )
        
        chem_loss_val = self.chem_loss(valid_mask, batch_size)
        
        # Knowledge consistency loss
        kg_loss_val = self.kg_loss(h_mol, h_kg)
        
        # Total loss: L_total = L_adv + α·L_prop + β·L_kg
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
    """
    Discriminator update step with KG enhancement.
    
    Key differences from Phase 1:
    - Get h_seq from ProteinEncoder
    - Get h_kg from KGConditionBuilder using protein_entity_indices
    - Fuse: h_fuse = GatedFusion(h_seq, h_kg)
    - Use h_fuse as condition for Generator and Discriminator
    """
    (protein, smiles, label, pro_mask, smi_mask, 
     raw_smiles, kg_entity_idx) = batch
    
    # Move tensors to device
    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    label = torch.as_tensor(label).to(device)
    kg_entity_idx = torch.as_tensor(kg_entity_idx).to(device)
    
    batch_size = protein.size(0)
    
    # Encode protein sequence
    protein_output = protein_encoder(protein, apply_mlm=False)
    h_seq = protein_output['protein_features']
    
    # Get KG embeddings and fuse
    if kg_condition_builder is not None and (kg_entity_idx >= 0).any():
        # Get KG embeddings for valid entities
        kg_embeddings = get_batch_kg_embeddings(
            kg_condition_builder.transe_model,
            kg_condition_builder.kg_dataset,
            kg_entity_idx,
            neighborhood_k=kg_condition_builder.neighborhood_k,
            device=device
        )
        
        # Project and fuse
        h_kg_projected = kg_condition_builder.projector(kg_embeddings)
        h_fuse, _ = kg_condition_builder.fusion(h_seq, h_kg_projected)
    else:
        h_fuse = h_seq
        h_kg_projected = None
    
    # Encode real SMILES
    real_mol_output = smi_encoder(smiles)
    real_mol_features = real_mol_output['token_features']
    
    # Discriminate real samples
    disc_real_output = discriminator(real_mol_features, h_fuse)
    
    # Generate fake samples
    if teacher_forcing:
        tgt_mask = torch.triu(
            torch.ones(smiles.size(1), smiles.size(1), device=device),
            diagonal=1
        ).masked_fill(torch.ones(smiles.size(1), smiles.size(1), device=device) == 1, float('-inf'))
        
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
    
    # Encode fake SMILES
    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    
    # Discriminate fake samples
    disc_fake_output = discriminator(fake_mol_features, h_fuse)
    
    # Compute properties of real molecules
    real_props = compute_mol_properties(raw_smiles)
    
    # Compute loss
    loss, loss_dict = joint_loss.discriminator_loss(
        d_real=disc_real_output['real_fake'],
        d_fake=disc_fake_output['real_fake'],
        pred_qed_real=disc_real_output['qed'],
        pred_sa_real=disc_real_output['sa'],
        pred_logp_real=disc_real_output['logp'],
        pred_affinity_real=disc_real_output.get('affinity', torch.zeros(batch_size, 1, device=h_fuse.device)),
        true_qed=real_props['qed'],
        true_sa=real_props['sa'],
        true_logp=real_props['logp'],
        true_affinity=real_props.get('affinity', [0.0] * batch_size)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    nn.utils.clip_grad_norm_(protein_encoder.parameters(), max_norm=1.0)
    nn.utils.clip_grad_norm_(smi_encoder.parameters(), max_norm=1.0)
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
    teacher_forcing: bool = True
) -> Dict[str, float]:
    """
    Generator update step with KG enhancement.
    """
    (protein, smiles, label, pro_mask, smi_mask, 
     raw_smiles, kg_entity_idx) = batch
    
    protein = torch.as_tensor(protein).to(device)
    smiles = torch.as_tensor(smiles).to(device)
    pro_mask = torch.as_tensor(pro_mask).to(device)
    smi_mask = torch.as_tensor(smi_mask).to(device)
    kg_entity_idx = torch.as_tensor(kg_entity_idx).to(device)
    
    batch_size = protein.size(0)
    
    # Encode protein and get KG-enhanced condition
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
    
    # Generate fake SMILES
    if teacher_forcing:
        gen_probs = F.softmax(generator(smiles, h_fuse), dim=-1)
        fake_smiles = torch.multinomial(gen_probs.view(-1, gen_probs.size(-1)), 1)
        fake_smiles = fake_smiles.view(batch_size, -1)
    else:
        start_token = smiVoc.index('&')
        fake_smiles = generator.generate(h_fuse, max_len=smiles.size(1), start_token=start_token)
    
    # Encode generated SMILES
    fake_mol_output = smi_encoder(fake_smiles)
    fake_mol_features = fake_mol_output['token_features']
    h_mol = fake_mol_output['molecular_features']
    
    # Discriminate
    disc_fake_output = discriminator(fake_mol_features, h_fuse)
    
    # Decode generated SMILES for validity check
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
    
    # Check validity
    valid_mask = []
    for smi in fake_smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            valid_mask.append(1 if mol is not None else 0)
        except:
            valid_mask.append(0)
    
    # Compute loss with KG consistency
    # Use h_kg_projected if available, otherwise use h_fuse as proxy
    h_kg_for_loss = h_kg_projected if h_kg_projected is not None else h_fuse
    
    loss, loss_dict = joint_loss.generator_loss(
        d_fake=disc_fake_output['real_fake'],
        pred_qed_fake=disc_fake_output['qed'],
        pred_sa_fake=disc_fake_output['sa'],
        pred_logp_fake=disc_fake_output['logp'],
        pred_affinity_fake=disc_fake_output.get('affinity', torch.zeros(batch_size, 1, device=h_fuse.device)),
        h_mol=h_mol,
        h_kg=h_kg_for_loss,
        valid_mask=valid_mask,
        batch_size=batch_size
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Add validity rate to metrics
    validity_rate = sum(valid_mask) / max(batch_size, 1)
    loss_dict['validity_rate'] = validity_rate
    
    return loss_dict


def train_kg_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    kg_condition_builder: Optional[KGConditionBuilder],
    train_loader: DataLoader,
    joint_loss: KGJointLoss,
    gen_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    device: torch.device,
    smiVoc: List[str],
    teacher_forcing: bool = True,
    n_critic: int = 1
) -> Dict[str, float]:
    """Train for one epoch with KG enhancement."""
    generator.train()
    discriminator.train()
    protein_encoder.train()
    smi_encoder.train()
    
    epoch_losses = {
        'gen_total': 0.0,
        'gen_gan': 0.0,
        'gen_bio': 0.0,
        'gen_chem': 0.0,
        'gen_kg': 0.0,
        'disc_total': 0.0,
        'disc_gan': 0.0,
        'disc_bio': 0.0,
        'validity_rate': 0.0
    }
    
    num_batches = len(train_loader)
    
    for batch in tqdm(train_loader, desc='Training'):
        batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
        
        # Discriminator updates
        for _ in range(n_critic):
            disc_losses = train_kg_discriminator_step(
                batch, protein_encoder, smi_encoder, generator, discriminator,
                kg_condition_builder, joint_loss, disc_optimizer, device, smiVoc, teacher_forcing
            )
            epoch_losses['disc_total'] += disc_losses['total_loss']
            epoch_losses['disc_gan'] += disc_losses['gan_loss']
            epoch_losses['disc_bio'] += disc_losses['bio_loss']
        
        # Generator update
        gen_losses = train_kg_generator_step(
            batch, protein_encoder, smi_encoder, generator, discriminator,
            kg_condition_builder, joint_loss, gen_optimizer, device, smiVoc, teacher_forcing
        )
        epoch_losses['gen_total'] += gen_losses['total_loss']
        epoch_losses['gen_gan'] += gen_losses['gan_loss']
        epoch_losses['gen_bio'] += gen_losses['bio_loss']
        epoch_losses['gen_chem'] += gen_losses['chem_loss']
        epoch_losses['gen_kg'] += gen_losses['kg_loss']
        epoch_losses['validity_rate'] += gen_losses.get('validity_rate', 0.0)
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


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
    """Validate for one epoch with KG enhancement."""
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
        'validity_rate': 0.0
    }
    
    num_batches = len(val_loader)
    
    for batch in tqdm(val_loader, desc='Validating'):
        batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
        (protein, smiles, label, pro_mask, smi_mask, 
         raw_smiles, kg_entity_idx) = batch
        
        protein = protein.to(device)
        smiles = smiles.to(device)
        kg_entity_idx = kg_entity_idx.to(device)
        batch_size = protein.size(0)
        
        # Encode protein
        protein_output = protein_encoder(protein, apply_mlm=False)
        h_seq = protein_output['protein_features']
        
        # Get KG-enhanced condition
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
        
        # Generate
        if teacher_forcing:
            gen_logits = generator(smiles, h_fuse)
            fake_smiles = gen_logits.argmax(dim=-1)
        else:
            start_token = smiVoc.index('&')
            fake_smiles = generator.generate(h_fuse, max_len=smiles.size(1), start_token=start_token)
        
        # Encode
        real_mol_output = smi_encoder(smiles)
        real_mol_features = real_mol_output['token_features']
        
        fake_mol_output = smi_encoder(fake_smiles)
        fake_mol_features = fake_mol_output['token_features']
        h_mol = fake_mol_output['molecular_features']
        
        # Discriminate
        disc_real_output = discriminator(real_mol_features, h_fuse)
        disc_fake_output = discriminator(fake_mol_features, h_fuse)
        
        # Properties
        real_props = compute_mol_properties(raw_smiles)
        
        # Discriminator loss
        disc_loss, _ = joint_loss.discriminator_loss(
            d_real=disc_real_output['real_fake'],
            d_fake=disc_fake_output['real_fake'],
            pred_qed_real=disc_real_output['qed'],
            pred_sa_real=disc_real_output['sa'],
            pred_logp_real=disc_real_output['logp'],
            pred_affinity_real=disc_real_output.get('affinity', torch.zeros(batch_size, 1, device=h_fuse.device)),
            true_qed=real_props['qed'],
            true_sa=real_props['sa'],
            true_logp=real_props['logp'],
            true_affinity=real_props.get('affinity', [0.0] * batch_size)
        )
        
        # Decode for validity
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
        
        # Generator loss
        h_kg_for_loss = h_kg_projected if h_kg_projected is not None else h_fuse
        gen_loss, gen_loss_dict = joint_loss.generator_loss(
            d_fake=disc_fake_output['real_fake'],
            pred_qed_fake=disc_fake_output['qed'],
            pred_sa_fake=disc_fake_output['sa'],
            pred_logp_fake=disc_fake_output['logp'],
            pred_affinity_fake=disc_fake_output.get('affinity', torch.zeros(batch_size, 1, device=h_fuse.device)),
            h_mol=h_mol,
            h_kg=h_kg_for_loss,
            valid_mask=valid_mask,
            batch_size=batch_size
        )
        
        # Accumulate
        val_losses['disc_total'] += disc_loss.item()
        val_losses['gen_total'] += gen_loss.item()
        val_losses['gen_gan'] += gen_loss_dict['gan_loss']
        val_losses['gen_bio'] += gen_loss_dict['bio_loss']
        val_losses['gen_chem'] += gen_loss_dict['chem_loss']
        val_losses['gen_kg'] += gen_loss_dict['kg_loss']
        val_losses['validity_rate'] += sum(valid_mask) / max(batch_size, 1)
    
    # Average
    for key in val_losses:
        val_losses[key] /= num_batches
    
    return val_losses


def save_kg_checkpoint(
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    protein_encoder: nn.Module,
    smi_encoder: nn.Module,
    kg_condition_builder: Optional[KGConditionBuilder],
    transe_model: Optional[TransEModel],
    gen_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    config: Dict[str, Any],
    output_dir: str
):
    """Save model checkpoint with KG state."""
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
    
    # Save KG-related state
    if kg_condition_builder is not None:
        checkpoint['kg_projector_state_dict'] = kg_condition_builder.projector.state_dict()
        checkpoint['gated_fusion_state_dict'] = kg_condition_builder.fusion.state_dict()
    
    if transe_model is not None:
        checkpoint['transe_state_dict'] = transe_model.state_dict()
    
    checkpoint_path = os.path.join(output_dir, 'model', f'epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def plot_kg_loss_curves(
    train_losses: List[Dict[str, float]],
    val_losses: List[Dict[str, float]],
    output_dir: str
):
    """Plot training and validation loss curves including KG loss."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Generator total loss
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
    
    # Discriminator total loss
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
    
    # Generator loss components
    ax = axes[0, 2]
    train_gen_gan = [loss['gen_gan'] for loss in train_losses]
    train_gen_bio = [loss['gen_bio'] for loss in train_losses]
    train_gen_kg = [loss['gen_kg'] for loss in train_losses]
    ax.plot(epochs, train_gen_gan, 'b-', label='GAN Loss')
    ax.plot(epochs, train_gen_bio, 'g-', label='Bio Loss')
    ax.plot(epochs, train_gen_kg, 'r-', label='KG Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Generator Loss Components')
    ax.legend()
    ax.grid(True)
    
    # Validity rate
    ax = axes[1, 0]
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
    
    # KG loss comparison
    ax = axes[1, 1]
    train_kg = [loss['gen_kg'] for loss in train_losses]
    val_kg = [loss['gen_kg'] for loss in val_losses]
    ax.plot(epochs, train_kg, 'b-', label='Train KG Loss')
    ax.plot(epochs, val_kg, 'r-', label='Val KG Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Knowledge Consistency Loss')
    ax.legend()
    ax.grid(True)
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logs', 'loss_curves.png'), dpi=150)
    plt.close()
    logger.info(f"Saved loss curves to {os.path.join(output_dir, 'logs', 'loss_curves.png')}")


def main():
    """Main training function for KG-enhanced GAN."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train KG-DrugGAN-MSM')
    
    # Base training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-critic', type=int, default=1, help='Number of critic updates per generator update')
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=512, help='Model hidden dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of protein encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=12, help='Number of generator decoder layers')
    parser.add_argument('--num-disc-layers', type=int, default=6, help='Number of discriminator encoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--noise-dim', type=int, default=756, help='Noise vector dimension')
    parser.add_argument('--mask-rate', type=float, default=0.15, help='MLM mask rate')
    
    # Loss weights
    parser.add_argument('--lambda-gan', type=float, default=1.0, help='GAN loss weight')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for property loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for knowledge consistency loss')
    parser.add_argument('--lambda-bio', type=float, default=0.5, help='Biology loss weight')
    parser.add_argument('--lambda-chem', type=float, default=0.3, help='Chemical validity loss weight')
    
    # KG-specific arguments
    parser.add_argument('--kg-embedding-dim', type=int, default=100, help='TransE embedding dimension')
    parser.add_argument('--kg-epochs', type=int, default=100, help='TransE pre-training epochs')
    parser.add_argument('--kg-lr', type=float, default=0.01, help='TransE learning rate')
    parser.add_argument('--kg-margin', type=float, default=1.0, help='TransE margin')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--kg-triples-path', type=str, default='./kg/kg_triples.csv', help='Path to KG triples CSV file')
    parser.add_argument('--kg-neighborhood-k', type=int, default=1, help='k-hop neighborhood size')
    parser.add_argument('--kg-topk', type=int, default=50, help='Max neighbors per relation type')
    parser.add_argument('--kg-pretrained-path', type=str, default='', help='Path to pre-trained TransE model')
    
    # General arguments
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--output-dir', type=str, default='./experiments/kg_druggan_msm', help='Output directory')
    parser.add_argument('--note', type=str, default='', help='Experiment note')
    parser.add_argument('--teacher-forcing', action='store_true', default=True, help='Use teacher forcing')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--skip-kg', action='store_true', help='Skip KG enhancement (run baseline)')
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
    
    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Setup logging
    logger.add(os.path.join(args.output_dir, 'logs', 'train.log'))
    logger.info(f"Training started at {datetime.now()}")
    logger.info(f"Arguments: {args}")
    
    # Load data config
    config = loadConfig(args)
    config.batchSize = args.batch_size
    
    # Save settings
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
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_bio': args.lambda_bio,
        'lambda_chem': args.lambda_chem,
        'n_critic': args.n_critic,
        'kg_embedding_dim': args.kg_embedding_dim,
        'kg_epochs': args.kg_epochs,
        'kg_lr': args.kg_lr,
        'kg_margin': args.kg_margin,
        'kg_neighborhood_k': args.kg_neighborhood_k
    }
    
    with open(os.path.join(args.output_dir, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)
    logger.info("Saved training settings")
    
    # Load or build KG
    kg_dataset = None
    transe_model = None
    kg_condition_builder = None
    
    if not args.skip_kg:
        # Prepare KG dataset
        if os.path.exists(args.kg_triples_path):
            kg_dataset = prepare_kg_dataset(args.kg_triples_path)
            
            # Build protein-to-KG mapping
            protein_to_kg_mapping = build_protein_to_kg_mapping(
                os.path.join(args.data_dir, 'train-val-data.tsv'),
                kg_dataset
            )
        else:
            logger.warning("KG triples not found. Running without KG enhancement.")
            args.skip_kg = True
            protein_to_kg_mapping = {}
    else:
        protein_to_kg_mapping = {}
    
    # Pre-train TransE if needed
    if not args.skip_kg and kg_dataset is not None:
        if args.kg_pretrained_path and os.path.exists(args.kg_pretrained_path):
            logger.info(f"Loading pre-trained TransE from {args.kg_pretrained_path}")
            checkpoint = torch.load(args.kg_pretrained_path, map_location=device)
            transe_model = TransEModel(
                num_entities=checkpoint.get('num_entities', kg_dataset.num_entities),
                num_relations=checkpoint.get('num_relations', kg_dataset.num_relations),
                embedding_dim=args.kg_embedding_dim
            )
            transe_model.load_state_dict(checkpoint['model_state_dict'])
            transe_model.to(device)
        else:
            # Pre-train TransE
            transe_save_path = os.path.join(args.output_dir, 'model', 'transe_best.pt')
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
        
        # Build KG condition builder
        kg_condition_builder = KGConditionBuilder(
            entity_embeddings=transe_model.get_all_entity_embeddings(),
            kg_adjacency={kg_dataset.entity_to_idx[e]: [kg_dataset.entity_to_idx[n] for n in neighbors]
                         for e, rels in kg_dataset.adjacency.items() for neighbors in rels.values()},
            d_model=args.d_model,
            kg_dim=args.kg_embedding_dim
        ).to(device)
    
    # Prepare datasets
    logger.info("Preparing training data...")
    train_data, train_smiles = prepareKGGANDataset(
        config, kg_dataset, protein_to_kg_mapping, 'train', data_dir=args.data_dir
    )
    val_data, val_smiles = prepareKGGANDataset(
        config, kg_dataset, protein_to_kg_mapping, 'valid', data_dir=args.data_dir
    )
    
    train_dataset = KGGANDataset(train_data, train_smiles, protein_to_kg_mapping, kg_dataset)
    val_dataset = KGGANDataset(val_data, val_smiles, protein_to_kg_mapping, kg_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize models
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
    
    # Import from DrugGAN_MSM
    from model.DrugGAN_MSM import Generator, MultiTaskDiscriminator
    
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
    
    protein_encoder = protein_encoder.to(device)
    smi_encoder = smi_encoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Setup optimizers
    gen_params = list(generator.parameters())
    disc_params = list(discriminator.parameters()) + list(protein_encoder.parameters()) + list(smi_encoder.parameters())
    
    # Add KG parameters if using KG
    if kg_condition_builder is not None:
        gen_params += list(kg_condition_builder.parameters())
    
    gen_optimizer = optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_params, lr=args.lr, betas=(0.5, 0.999))
    
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=args.epochs, eta_min=1e-6)
    disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Setup loss function
    joint_loss = KGJointLoss(
        lambda_gan=args.lambda_gan,
        alpha=args.alpha,
        beta=args.beta,
        lambda_bio=args.lambda_bio,
        lambda_chem=args.lambda_chem
    )
    
    # Resume from checkpoint if specified
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
            
            if 'kg_projector_state_dict' in checkpoint and kg_condition_builder is not None:
                kg_condition_builder.projector.load_state_dict(checkpoint['kg_projector_state_dict'])
            if 'gated_fusion_state_dict' in checkpoint and kg_condition_builder is not None:
                kg_condition_builder.fusion.load_state_dict(checkpoint['gated_fusion_state_dict'])
            
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    
    # Training loop
    train_history = []
    val_history = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_losses = train_kg_epoch(
            generator, discriminator, protein_encoder, smi_encoder,
            kg_condition_builder, train_loader, joint_loss,
            gen_optimizer, disc_optimizer, device, config.smiVoc,
            teacher_forcing=args.teacher_forcing, n_critic=args.n_critic
        )
        
        # Validate
        val_losses = validate_kg_epoch(
            generator, discriminator, protein_encoder, smi_encoder,
            kg_condition_builder, val_loader, joint_loss,
            device, config.smiVoc, teacher_forcing=args.teacher_forcing
        )
        
        # Log
        logger.info(f"Train - Gen Total: {train_losses['gen_total']:.4f}, "
                   f"Disc Total: {train_losses['disc_total']:.4f}, "
                   f"KG Loss: {train_losses['gen_kg']:.4f}, "
                   f"Validity: {train_losses['validity_rate']:.4f}")
        logger.info(f"Val   - Gen Total: {val_losses['gen_total']:.4f}, "
                   f"Disc Total: {val_losses['disc_total']:.4f}, "
                   f"KG Loss: {val_losses['gen_kg']:.4f}, "
                   f"Validity: {val_losses['validity_rate']:.4f}")
        
        # Save history
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
            save_kg_checkpoint(
                epoch, generator, discriminator, protein_encoder, smi_encoder,
                kg_condition_builder, transe_model,
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
            save_kg_checkpoint(
                epoch, generator, discriminator, protein_encoder, smi_encoder,
                kg_condition_builder, transe_model,
                gen_optimizer, disc_optimizer, settings, args.output_dir
            )
        
        # Plot loss curves
        plot_kg_loss_curves(train_history, val_history, args.output_dir)
    
    # Save final checkpoint
    save_kg_checkpoint(
        args.epochs - 1, generator, discriminator, protein_encoder, smi_encoder,
        kg_condition_builder, transe_model,
        gen_optimizer, disc_optimizer, settings, args.output_dir
    )
    
    # Log completion
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logger.info(f"\nTraining completed in {total_time:.2f} hours")
    logger.info(f"Final model saved to {os.path.join(args.output_dir, 'model', f'epoch_{args.epochs - 1}.pt')}")


if __name__ == '__main__':
    main()