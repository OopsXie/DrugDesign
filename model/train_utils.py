"""
Shared Training Utilities for DrugGAN-MSM

This module contains shared loss functions and utility functions used across
training scripts to avoid code duplication.

Author: DrugGAN-MSM Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from utils.sascorer import calculateScore as compute_sa_score


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class GANLoss(nn.Module):
    """
    GAN adversarial loss for discriminator and generator.
    
    Implements binary cross-entropy loss for:
    - Discriminator: distinguish real vs fake samples
    - Generator: fool the discriminator
    """
    
    def __init__(self):
        super(GANLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward_discriminator(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            d_real: Discriminator output for real samples
            d_fake: Discriminator output for fake samples
            
        Returns:
            Total discriminator loss (real + fake)
        """
        real_targets = torch.ones_like(d_real)
        real_loss = self.bce_loss(d_real, real_targets)
        
        fake_targets = torch.zeros_like(d_fake)
        fake_loss = self.bce_loss(d_fake, fake_targets)
        
        return real_loss + fake_loss
    
    def forward_generator(self, d_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            d_fake: Discriminator output for fake samples
            
        Returns:
            Generator loss (trying to make discriminator predict 'real')
        """
        real_targets = torch.ones_like(d_fake)
        return self.bce_loss(d_fake, real_targets)


class BiologyPropertyLoss(nn.Module):
    """
    Biology property loss for QED, SA, logP, and affinity prediction.
    
    Computes MSE loss between predicted and target property values.
    """
    
    def __init__(self):
        super(BiologyPropertyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        pred_qed: torch.Tensor,
        pred_sa: torch.Tensor,
        pred_logp: torch.Tensor,
        pred_affinity: torch.Tensor,
        true_qed: List[float],
        true_sa: List[float],
        true_logp: List[float],
        true_affinity: List[float]
    ) -> torch.Tensor:
        """
        Compute biology property loss.
        
        Args:
            pred_qed: Predicted QED scores
            pred_sa: Predicted SA scores
            pred_logp: Predicted logP values
            pred_affinity: Predicted affinity values
            true_qed: Ground truth QED scores
            true_sa: Ground truth SA scores
            true_logp: Ground truth logP values
            true_affinity: Ground truth affinity values
            
        Returns:
            Total property loss (QED + SA + logP + affinity)
        """
        device = pred_qed.device
        true_qed_tensor = torch.tensor(true_qed, dtype=torch.float32, device=device).unsqueeze(1)
        true_sa_tensor = torch.tensor(true_sa, dtype=torch.float32, device=device).unsqueeze(1)
        true_logp_tensor = torch.tensor(true_logp, dtype=torch.float32, device=device).unsqueeze(1)
        true_affinity_tensor = torch.tensor(true_affinity, dtype=torch.float32, device=device).unsqueeze(1)
        
        qed_loss = self.mse_loss(pred_qed, true_qed_tensor)
        sa_loss = self.mse_loss(pred_sa, true_sa_tensor)
        logp_loss = self.mse_loss(pred_logp, true_logp_tensor)
        affinity_loss = self.mse_loss(pred_affinity, true_affinity_tensor)
        
        return qed_loss + sa_loss + logp_loss + affinity_loss


class ChemicalValidityLoss(nn.Module):
    """
    Chemical validity loss based on molecule validity rate.
    
    Encourages the generator to produce valid SMILES strings.
    """
    
    def __init__(self):
        super(ChemicalValidityLoss, self).__init__()
    
    def forward(self, valid_mask: List[int], batch_size: int) -> torch.Tensor:
        """
        Compute chemical validity loss.
        
        Args:
            valid_mask: Binary mask indicating valid molecules (1=valid, 0=invalid)
            batch_size: Batch size
            
        Returns:
            Validity loss (negative validity rate to maximize validity)
        """
        valid_count = sum(valid_mask)
        validity_rate = valid_count / max(batch_size, 1)
        loss = -validity_rate * batch_size
        return torch.tensor(loss, dtype=torch.float32)


def compute_mol_properties(smiles_list: List[str]) -> Dict[str, List[float]]:
    """
    Compute RDKit-based property values for a batch of SMILES strings.
    
    These properties serve as "ground truth" for the biology property loss.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Dictionary containing:
        - 'qed': QED scores (0-1, higher is better)
        - 'sa': Synthetic accessibility scores (1-10, lower is easier to synthesize)
        - 'logp': LogP values (lipophilicity)
        - 'affinity': Binding affinity proxy (kcal/mol, lower is better)
        - 'valid_mask': Binary mask indicating valid molecules (1=valid, 0=invalid)
    """
    qed_values = []
    sa_values = []
    logp_values = []
    affinity_values = []
    valid_masks = []
    
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol = Chem.RemoveHs(mol, sanitize=False)
                qed = QED.qed(mol)
                sa = compute_sa_score(mol)
                logp = Descriptors.MolLogP(mol)
                mol_weight = Descriptors.MolWt(mol)
                
                # Affinity proxy: simplified binding energy approximation
                polar_surface_area = Descriptors.TPSA(mol)
                n_h_donors = Descriptors.NumHDonors(mol)
                n_h_acceptors = Descriptors.NumHAcceptors(mol)
                hbond_contrib = -0.03 * (n_h_donors + n_h_acceptors)
                affinity = -0.5 * logp - 0.005 * mol_weight - 0.003 * polar_surface_area + hbond_contrib
                
                qed_values.append(qed)
                sa_values.append(sa)
                logp_values.append(logp)
                affinity_values.append(affinity)
                valid_masks.append(1)
            else:
                qed_values.append(0.0)
                sa_values.append(10.0)
                logp_values.append(0.0)
                affinity_values.append(0.0)
                valid_masks.append(0)
        except Exception:
            qed_values.append(0.0)
            sa_values.append(10.0)
            logp_values.append(0.0)
            affinity_values.append(0.0)
            valid_masks.append(0)
    
    return {
        'qed': qed_values,
        'sa': sa_values,
        'logp': logp_values,
        'affinity': affinity_values,
        'valid_mask': valid_masks
    }


def compute_uniqueness(smiles_list: List[str]) -> float:
    """
    Compute uniqueness metric for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Uniqueness ratio (unique_smiles / total_smiles)
    """
    if len(smiles_list) == 0:
        return 0.0

    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list)


def compute_validity(generated_smiles: List[str]) -> float:
    """
    Compute validity ratio: fraction of SMILES that RDKit can parse.

    Args:
        generated_smiles: List of generated SMILES strings

    Returns:
        Validity ratio in [0, 1]
    """
    if len(generated_smiles) == 0:
        return 0.0

    valid_count = 0
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_count += 1
    return valid_count / len(generated_smiles)


def compute_novelty(valid_smiles: List[str], training_smiles: set) -> float:
    """
    Compute novelty ratio: fraction of valid SMILES not seen in training set.

    Args:
        valid_smiles: List of valid canonical SMILES
        training_smiles: Set of training SMILES for comparison

    Returns:
        Novelty ratio in [0, 1]
    """
    if len(valid_smiles) == 0:
        return 0.0

    novel_count = 0
    for smi in valid_smiles:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True) if Chem.MolFromSmiles(smi) else smi
        if canonical not in training_smiles:
            novel_count += 1
    return novel_count / len(valid_smiles)


def compute_all_metrics(
    generated_smiles: List[str],
    training_smiles: Optional[set] = None,
) -> Dict[str, float]:
    """
    Compute all generation quality metrics in one call.

    Args:
        generated_smiles: List of generated SMILES strings
        training_smiles: Optional set of training SMILES for novelty computation

    Returns:
        Dictionary with keys: validity, uniqueness, novelty, mean_qed, mean_sa, mean_logp
    """
    # Validity
    valid_mols = []
    valid_smiles = []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(smi)

    total = len(generated_smiles)
    validity = len(valid_mols) / total if total > 0 else 0.0

    # Uniqueness (among valid)
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles) if len(valid_smiles) > 0 else 0.0

    # Novelty
    if training_smiles is not None and len(valid_smiles) > 0:
        novel_count = 0
        for smi in unique_smiles:
            if smi not in training_smiles:
                novel_count += 1
        novelty = novel_count / len(unique_smiles)
    else:
        novelty = -1.0  # Not computed

    # Property averages (among valid molecules)
    qed_vals, sa_vals, logp_vals = [], [], []
    for mol in valid_mols:
        qed_vals.append(QED.qed(mol))
        sa_vals.append(compute_sa_score(mol))
        logp_vals.append(Descriptors.MolLogP(mol))

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'mean_qed': float(np.mean(qed_vals)) if qed_vals else 0.0,
        'mean_sa': float(np.mean(sa_vals)) if sa_vals else 0.0,
        'mean_logp': float(np.mean(logp_vals)) if logp_vals else 0.0,
        'num_valid': len(valid_mols),
        'num_unique': len(unique_smiles),
        'num_total': total,
    }