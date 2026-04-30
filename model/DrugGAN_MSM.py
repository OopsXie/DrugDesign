"""
DrugGAN-MSM Model Architecture

This module implements the conditional GAN architecture for drug discovery
as described in Chapter 3 of the thesis. It includes:
- ProteinEncoder: Encodes protein amino acid sequences with MLM and self-attention
- SMILESEncoder: Encodes SMILES molecular representations
- Generator: Conditional generator for SMILES sequences given protein features
- MultiTaskDiscriminator: Discriminator with real/fake and property prediction heads
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Implements sinusoidal positional encodings as described in 
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    Args:
        d_model: Dimension of the model embeddings
        dropout: Dropout probability
        max_len: Maximum sequence length to encode
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer so it's not treated as a parameter
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model) or (seq_len, batch, d_model)
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        x = x + self.pe
        return self.dropout(x)


class ProteinEncoder(nn.Module):
    """
    Protein sequence encoder with MLM and self-attention modules.
    
    Section 3.2.1 of the thesis. Encodes protein amino acid sequences into
    feature representations using:
    1. Token embedding with positional encoding
    2. MLM module: randomly masks 15% of amino acids and predicts them
    3. Self-attention module: 12-layer TransformerEncoder
    4. Weighted fusion: combines MLM and self-attention outputs with learnable weights
    
    Args:
        pro_voc_len: Size of protein vocabulary (number of unique amino acid tokens)
        d_model: Dimension of model embeddings (default: 512)
        mask_rate: Rate of masking for MLM (default: 0.15 = 15%)
        mask_token_id: Token ID for [MASK] token (default: 3)
    """
    
    def __init__(
        self,
        pro_voc_len: int,
        d_model: int = 512,
        mask_rate: float = 0.15,
        mask_token_id: int = 3,
        padding_idx: int = 0
    ):
        super(ProteinEncoder, self).__init__()
        
        self.d_model = d_model
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id
        self.padding_idx = padding_idx
        
        # Token embedding
        self.embedding = nn.Embedding(pro_voc_len, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        
        # MLM prediction head: projects from d_model to vocabulary size
        self.mlm_head = nn.Linear(d_model, pro_voc_len)
        
        # Self-attention module: 12-layer TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,  # 4 * d_model as per standard
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=12,
            norm=nn.LayerNorm(d_model)
        )
        
        # Weighted fusion parameters (learnable scalars)
        # Formula 3-8: Z_P = W_m * Z_MLM + W_s * Z_SelfAttention
        self.W_m = nn.Parameter(torch.tensor(1.0))
        self.W_s = nn.Parameter(torch.tensor(1.0))
        
        # LayerNorm for fusion output (Formula 3-9)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # MLP for Z_MLM processing (independent from embedding output)
        # Gives Z_MLM a proper learned transformation instead of just raw embedding
        self.mlm_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Output projection to 756 dimensions
        self.output_proj = nn.Linear(d_model, 756)
    
    def apply_mask(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking to protein tokens for MLM training.
        
        Args:
            tokens: Input token indices of shape (batch, seq_len)
        
        Returns:
            Tuple of (masked_tokens, mask_positions):
                - masked_tokens: tokens with mask_rate% replaced by [MASK]
                - mask_positions: boolean mask indicating which positions were masked
        """
        # Create a mask for positions to mask (excluding padding)
        non_padding_mask = (tokens != self.padding_idx)
        mask_shape = tokens.shape
        rand_mask = torch.rand(mask_shape, device=tokens.device) < self.mask_rate
        
        # Only mask non-padding positions
        mask = rand_mask & non_padding_mask
        
        # Create masked tokens
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        return masked_tokens, mask
    
    def forward(
        self,
        tokens: torch.Tensor,
        apply_mlm: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode protein sequence into feature representation.
        
        Args:
            tokens: Protein token indices of shape (batch, seq_len)
            apply_mlm: Whether to apply MLM masking (True for training, False for inference)
        
        Returns:
            Dictionary containing:
                - 'protein_features': Pooled protein feature vector (batch, 756)
                - 'token_features': Per-token features (batch, seq_len, d_model)
                - 'mlm_logits': MLM prediction logits (batch, seq_len, pro_voc_len)
                - 'mask': Boolean mask of MLM-masked positions (only if apply_mlm=True)
        """
        batch_size, seq_len = tokens.shape
        
        # Apply MLM masking if training
        if apply_mlm:
            masked_tokens, mask = self.apply_mask(tokens)
        else:
            masked_tokens = tokens
            mask = None
        
        # Embed tokens
        x = self.embedding(masked_tokens)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Get MLM predictions from embedding level (before transformer)
        # NOTE: MLM head predicts from pre-transformer embeddings (BERT-style architecture)
        # Z_MLM's MLP (mlm_mlp) is for feature fusion, NOT for MLM loss
        mlm_logits = self.mlm_head(x)  # (batch, seq_len, pro_voc_len)
        z_mlm = self.mlm_mlp(x)  # Process through MLP for learned transformation (batch, seq_len, d_model)
        
        # Pass through Transformer encoder for self-attention features
        z_self = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Weighted fusion (Formula 3-8, 3-9)
        # Z_P = LayerNorm(W_m * Z_MLM + W_s * Z_SelfAttention)
        z_fused = self.W_m * z_mlm + self.W_s * z_self
        z_fused = self.fusion_norm(z_fused)  # (batch, seq_len, d_model)
        
        # Pool to get single protein feature vector (mean pooling)
        protein_features = z_fused.mean(dim=1)  # (batch, d_model)
        protein_features = self.output_proj(protein_features)  # (batch, 756)
        
        result = {
            'protein_features': protein_features,
            'token_features': z_fused,
            'mlm_logits': mlm_logits
        }
        
        if mask is not None:
            result['mask'] = mask
        
        return result


class SMILESEncoder(nn.Module):
    """
    SMILES molecular sequence encoder with MLM and self-attention modules.
    
    Section 3.2.2 of the thesis. Encodes SMILES token sequences into
    molecular feature representations using:
    1. Token embedding with positional encoding
    2. MLM module: randomly masks 15% of SMILES tokens and predicts them
    3. Self-attention module: 12-layer TransformerEncoder
    4. Weighted fusion: combines MLM and self-attention outputs with learnable weights
    
    Args:
        smi_voc_len: Size of SMILES vocabulary (number of unique tokens)
        d_model: Dimension of model embeddings (default: 512)
        mask_rate: Rate of masking for MLM (default: 0.15 = 15%)
        mask_token_id: Token ID for [MASK] token (default: 3)
        padding_idx: Token ID for padding (default: 0)
    """
    
    def __init__(
        self,
        smi_voc_len: int,
        d_model: int = 512,
        mask_rate: float = 0.15,
        mask_token_id: int = 3,
        padding_idx: int = 0
    ):
        super(SMILESEncoder, self).__init__()
        
        self.d_model = d_model
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id
        self.padding_idx = padding_idx
        
        # Token embedding
        self.embedding = nn.Embedding(smi_voc_len, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        
        # MLM prediction head: projects from d_model to vocabulary size
        self.mlm_head = nn.Linear(d_model, smi_voc_len)
        
        # Self-attention module: 12-layer TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=12,
            norm=nn.LayerNorm(d_model)
        )
        
        # Weighted fusion parameters (learnable scalars)
        # Formula 3-8: Z_M = W_m * Z_MLM + W_s * Z_SelfAttention
        self.W_m = nn.Parameter(torch.tensor(1.0))
        self.W_s = nn.Parameter(torch.tensor(1.0))
        
        # LayerNorm for fusion output (Formula 3-9)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # MLP for Z_MLM processing (independent from embedding output)
        # Gives Z_MLM a proper learned transformation instead of just raw embedding
        self.mlm_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Output projection to 756 dimensions
        self.output_proj = nn.Linear(d_model, 756)
        
        # SMILES token prediction head (for reconstruction/validation)
        self.token_head = nn.Linear(d_model, smi_voc_len)
    
    def apply_mask(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking to SMILES tokens for MLM training.
        
        Args:
            tokens: Input token indices of shape (batch, seq_len)
        
        Returns:
            Tuple of (masked_tokens, mask_positions):
                - masked_tokens: tokens with mask_rate% replaced by [MASK]
                - mask_positions: boolean mask indicating which positions were masked
        """
        # Create a mask for positions to mask (excluding padding)
        non_padding_mask = (tokens != self.padding_idx)
        mask_shape = tokens.shape
        rand_mask = torch.rand(mask_shape, device=tokens.device) < self.mask_rate
        
        # Only mask non-padding positions
        mask = rand_mask & non_padding_mask
        
        # Create masked tokens
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        return masked_tokens, mask
    
    def forward(
        self,
        tokens: torch.Tensor,
        apply_mlm: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode SMILES sequence into molecular feature representation.
        
        Args:
            tokens: SMILES token indices of shape (batch, seq_len)
            apply_mlm: Whether to apply MLM masking (True for training, False for inference)
        
        Returns:
            Dictionary containing:
                - 'molecular_features': Pooled molecular feature vector (batch, 756)
                - 'token_features': Per-token features (batch, seq_len, d_model)
                - 'token_logits': SMILES token prediction logits (batch, seq_len, smi_voc_len)
                - 'smi_mlm_logits': MLM prediction logits (batch, seq_len, smi_voc_len)
                - 'mask': Boolean mask of MLM-masked positions (only if apply_mlm=True)
        """
        batch_size, seq_len = tokens.shape
        
        # Apply MLM masking if training
        if apply_mlm:
            masked_tokens, mask = self.apply_mask(tokens)
        else:
            masked_tokens = tokens
            mask = None
        
        # Embed tokens
        x = self.embedding(masked_tokens)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Get MLM predictions from embedding level (before transformer)
        smi_mlm_logits = self.mlm_head(x)  # (batch, seq_len, smi_voc_len)
        z_mlm = self.mlm_mlp(x)  # Process through MLP for learned transformation (batch, seq_len, d_model)
        
        # Pass through Transformer encoder for self-attention features
        z_self = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Weighted fusion (Formula 3-8, 3-9)
        # Z_M = LayerNorm(W_m * Z_MLM + W_s * Z_SelfAttention)
        z_fused = self.W_m * z_mlm + self.W_s * z_self
        z_fused = self.fusion_norm(z_fused)  # (batch, seq_len, d_model)
        
        # Pool to get single molecular feature vector (mean pooling)
        molecular_features = z_fused.mean(dim=1)  # (batch, d_model)
        molecular_features = self.output_proj(molecular_features)  # (batch, 756)
        
        # Get token logits
        token_logits = self.token_head(z_fused)  # (batch, seq_len, smi_voc_len)
        
        result = {
            'molecular_features': molecular_features,
            'token_features': z_fused,
            'token_logits': token_logits,
            'smi_mlm_logits': smi_mlm_logits
        }
        
        if mask is not None:
            result['mask'] = mask
        
        return result


class Generator(nn.Module):
    """
    Conditional Generator for SMILES sequence generation.
    
    Section 3.3, Figure 3.4 of the thesis. Generates SMILES sequences
    conditioned on protein features using a TransformerDecoder.
    
    The generator takes:
    - A noise vector z ~ N(0, I)
    - Protein condition features Z_p from ProteinEncoder (756-dim)
    - Partial SMILES tokens for autoregressive generation
    
    Args:
        smi_voc_len: Size of SMILES vocabulary
        d_model: Dimension of model embeddings (default: 512)
        noise_dim: Dimension of input noise vector (default: 128)
        padding_idx: Token ID for padding (default: 0)
        num_layers: Number of decoder layers (default: 12)
        protein_dim: Dimension of protein input features (default: 756)
    """
    
    def __init__(
        self,
        smi_voc_len: int,
        d_model: int = 512,
        noise_dim: int = 128,
        padding_idx: int = 0,
        num_layers: int = 12,
        protein_dim: int = 756
    ):
        super(Generator, self).__init__()
        
        self.d_model = d_model
        self.noise_dim = noise_dim
        self.padding_idx = padding_idx
        
        # SMILES token embedding
        self.embedding = nn.Embedding(smi_voc_len, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        
        # Project noise vector to d_model dimension
        self.noise_projector = nn.Linear(noise_dim, d_model)
        
        # Project protein features from 756-dim to d_model
        self.protein_proj = nn.Linear(protein_dim, d_model)
        
        # Transformer decoder for conditional generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection to vocabulary
        self.token_head = nn.Linear(d_model, smi_voc_len)
    
    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal (triangular) mask for autoregressive generation.
        
        Args:
            seq_len: Length of sequence
            device: Device to create mask on
        
        Returns:
            Causal mask of shape (seq_len, seq_len) with -inf for future positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        partial_smiles_tokens: torch.Tensor,
        protein_features: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate SMILES token logits conditioned on protein features.
        
        Args:
            partial_smiles_tokens: Current SMILES sequence being generated,
                                   shape (batch, seq_len)
            protein_features: Protein condition features from ProteinEncoder,
                             shape (batch, 756)
            z: Optional noise vector, shape (batch, noise_dim).
               If None, zeros are used.
        
        Returns:
            SMILES token logits (raw, unnormalized) of shape (batch, seq_len, smi_voc_len).
            Apply F.softmax(logits, dim=-1) to get probabilities, or logits.argmax(dim=-1) for token indices.
        """
        batch_size, seq_len = partial_smiles_tokens.shape
        device = partial_smiles_tokens.device
        
        # Handle noise vector
        if z is None:
            z = torch.zeros(batch_size, self.noise_dim, device=device)
        
        # Project noise to d_model
        z_proj = self.noise_projector(z)  # (batch, d_model)
        
        # Project protein features from 756-dim to d_model
        protein_features = self.protein_proj(protein_features)  # (batch, d_model)
        
        # Expand protein features to sequence form for memory
        # Z_p_expanded = Z_p.unsqueeze(1).repeat(1, seq_len, 1)
        # But we also inject noise by adding it to the expanded protein features
        protein_memory = protein_features.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, d_model)
        
        # Combine noise with protein features (add noise projection to each position)
        protein_memory = protein_memory + z_proj.unsqueeze(1)  # (batch, seq_len, d_model)
        
        # Embed target tokens
        tgt = self.embedding(partial_smiles_tokens)  # (batch, seq_len, d_model)
        tgt = self.pos_encoder(tgt)
        
        # Create causal mask for autoregressive generation
        tgt_mask = self.generate_causal_mask(seq_len, device)
        
        # Pass through transformer decoder
        # tgt: SMILES embeddings, memory: protein condition
        output = self.transformer_decoder(
            tgt=tgt,
            memory=protein_memory,
            tgt_mask=tgt_mask
        )  # (batch, seq_len, d_model)
        
        # Project to vocabulary to get logits
        logits = self.token_head(output)  # (batch, seq_len, smi_voc_len)
        
        return logits


class MultiTaskDiscriminator(nn.Module):
    """
    Multi-task Discriminator with real/fake classification and property prediction.
    
    Section 3.3, Figure 3.5 of the thesis. The discriminator:
    1. Fuses molecular features X_M and protein features Z_p
    2. Performs real/fake classification (adversarial loss)
    3. Predicts 4 molecular properties: affinity, QED, SA, logP
    
    Args:
        protein_dim: Dimension of protein/molecular input features (default: 756)
        d_model: Legacy parameter, kept for backward compatibility (default: 512, unused if protein_dim specified)
        hidden_dim: Hidden dimension for property embedding (default: 256)
    """
    
    def __init__(
        self,
        protein_dim: int = 756,
        d_model: int = 512,
        hidden_dim: int = 256
    ):
        super(MultiTaskDiscriminator, self).__init__()
        
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        # Fusion layer: concatenates protein and molecular features
        # concat = ReLU(H_Zp ⊕ H_XM) where ⊕ is concatenation
        # Input dim: protein_dim (Z_p) + protein_dim (pooled X_M) = 2 * protein_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(2 * protein_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Real/Fake classification (Formula 3-12)
        # D_score = Sigmoid(W_D · H_XM + W_P · Z_p + b)
        # Uses concatenated features (protein + molecular) for proper fusion
        # Input is concat_features which has dimension hidden_dim after fusion_fc
        self.d_real_fake = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Property prediction embedding
        self.property_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Four property prediction heads (each outputs scalar)
        # 1. Affinity proxy (binding affinity)
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 2. QED (Quantitative Estimation of Drug-likeness)
        self.qed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. SA (Synthetic Accessibility)
        self.sa_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 4. logP (lipophilicity)
        self.logp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        molecular_token_features: torch.Tensor,
        protein_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Discriminate real/fake and predict molecular properties.
        
        Args:
            molecular_token_features: Per-token molecular features from SMILESEncoder,
                                     shape (batch, seq_len, protein_dim)
            protein_features: Protein features from ProteinEncoder,
                             shape (batch, protein_dim)
        
        Returns:
            Dictionary containing:
                - 'real_fake': Real/fake classification score (batch, 1)
                - 'affinity': Predicted binding affinity (batch, 1)
                - 'qed': Predicted QED score (batch, 1)
                - 'sa': Predicted synthetic accessibility (batch, 1)
                - 'logp': Predicted logP (batch, 1)
        """
        # Pool molecular features to single vector (mean pooling)
        molecular_features = molecular_token_features.mean(dim=1)  # (batch, protein_dim)
        
        # Concatenate protein and molecular features (Formula 3-10)
        # H_Zp = Z_p, H_XM = pooled(X_M)
        # concat = ReLU(H_Zp ⊕ H_XM)
        concat_features = torch.cat([protein_features, molecular_features], dim=1)
        concat_features = self.fusion_fc(concat_features)  # (batch, hidden_dim)
        
        # Real/Fake classification (Formula 3-12)
        # D(X_M, Z_p) = σ(W_D · X_M + W_P · Z_p + b)
        # Use concatenated features (protein + molecular) for proper fusion as per paper
        real_fake_score = self.d_real_fake(concat_features)  # (batch, 1)
        
        # Property prediction through embedding
        property_features = self.property_embedding(concat_features)
        
        # Four property prediction heads (Formula 3-10, 3-11)
        pred_affinity = self.affinity_head(property_features)  # (batch, 1)
        pred_qed = self.qed_head(property_features)  # (batch, 1)
        pred_sa = self.sa_head(property_features)  # (batch, 1)
        pred_logp = self.logp_head(property_features)  # (batch, 1)
        
        return {
            'real_fake': real_fake_score,
            'affinity': pred_affinity,
            'qed': pred_qed,
            'sa': pred_sa,
            'logp': pred_logp
        }


class ProteinEncoderNoMLM(ProteinEncoder):
    """
    Ablation variant: ProteinEncoder without MLM module.
    
    Uses only self-attention features (Z_SelfAttention) for fusion.
    MLM logits return zeros and are not used.
    """
    
    def __init__(self, pro_voc_len: int, d_model: int = 512, **kwargs):
        super().__init__(pro_voc_len, d_model, **kwargs)
    
    def forward(self, tokens: torch.Tensor, apply_mlm: bool = True) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = tokens.shape
        
        # No masking - use tokens directly
        masked_tokens = tokens
        
        # Embed tokens
        x = self.embedding(masked_tokens)
        x = self.pos_encoder(x)
        
        # MLM logits are zeros (not used in this variant)
        mlm_logits = torch.zeros(batch_size, seq_len, self.mlm_head.out_features, device=x.device)
        
        # Z_MLM is zeros - fusion will use only Z_SelfAttention
        z_mlm = torch.zeros_like(x)
        
        # Pass through Transformer encoder for self-attention features
        z_self = self.transformer_encoder(x)
        
        # Weighted fusion - only Z_SelfAttention contributes
        z_fused = self.W_m * z_mlm + self.W_s * z_self
        z_fused = self.fusion_norm(z_fused)
        
        # Pool to get protein feature vector
        protein_features = z_fused.mean(dim=1)
        protein_features = self.output_proj(protein_features)
        
        return {
            'protein_features': protein_features,
            'token_features': z_fused,
            'mlm_logits': mlm_logits
        }


class ProteinEncoderNoSA(ProteinEncoder):
    """
    Ablation variant: ProteinEncoder without self-attention module.
    
    Uses only MLM features (Z_MLM after mlm_mlp) for fusion.
    Transformer encoder is skipped.
    """
    
    def __init__(self, pro_voc_len: int, d_model: int = 512, **kwargs):
        super().__init__(pro_voc_len, d_model, **kwargs)
    
    def forward(self, tokens: torch.Tensor, apply_mlm: bool = True) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = tokens.shape
        
        # Apply MLM masking if training
        if apply_mlm:
            masked_tokens, mask = self.apply_mask(tokens)
        else:
            masked_tokens = tokens
            mask = None
        
        # Embed tokens
        x = self.embedding(masked_tokens)
        x = self.pos_encoder(x)
        
        # Get MLM predictions
        mlm_logits = self.mlm_head(x)
        z_mlm = self.mlm_mlp(x)
        
        # Skip transformer - Z_SelfAttention is zeros
        z_self = torch.zeros_like(x)
        
        # Weighted fusion - only Z_MLM contributes
        z_fused = self.W_m * z_mlm + self.W_s * z_self
        z_fused = self.fusion_norm(z_fused)
        
        # Pool to get protein feature vector
        protein_features = z_fused.mean(dim=1)
        protein_features = self.output_proj(protein_features)
        
        result = {
            'protein_features': protein_features,
            'token_features': z_fused,
            'mlm_logits': mlm_logits
        }
        
        if mask is not None:
            result['mask'] = mask
        
        return result


class SMILESEncoderNoMLM(SMILESEncoder):
    """
    Ablation variant: SMILESEncoder without MLM module.
    
    Uses only self-attention features for fusion.
    MLM logits return zeros and are not used.
    """
    
    def __init__(self, smi_voc_len: int, d_model: int = 512, **kwargs):
        super().__init__(smi_voc_len, d_model, **kwargs)
    
    def forward(self, tokens: torch.Tensor, apply_mlm: bool = True) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = tokens.shape
        
        # No masking - use tokens directly
        masked_tokens = tokens
        
        # Embed tokens
        x = self.embedding(masked_tokens)
        x = self.pos_encoder(x)
        
        # MLM logits are zeros (not used in this variant)
        smi_mlm_logits = torch.zeros(batch_size, seq_len, self.mlm_head.out_features, device=x.device)
        
        # Z_MLM is zeros - fusion will use only Z_SelfAttention
        z_mlm = torch.zeros_like(x)
        
        # Pass through Transformer encoder for self-attention features
        z_self = self.transformer_encoder(x)
        
        # Weighted fusion
        z_fused = self.W_m * z_mlm + self.W_s * z_self
        z_fused = self.fusion_norm(z_fused)
        
        # Pool to get molecular feature vector
        molecular_features = z_fused.mean(dim=1)
        molecular_features = self.output_proj(molecular_features)
        
        # Get token logits
        token_logits = self.token_head(z_fused)
        
        return {
            'molecular_features': molecular_features,
            'token_features': z_fused,
            'token_logits': token_logits,
            'smi_mlm_logits': smi_mlm_logits
        }


class SimpleDiscriminator(nn.Module):
    """
    Ablation variant: Simplified discriminator with real/fake classification only.
    
    No property prediction heads (affinity, QED, SA, logP).
    Used to study the impact of multi-task learning on generator performance.
    """
    
    def __init__(self, protein_dim: int = 756, d_model: int = 512, hidden_dim: int = 256):
        super(SimpleDiscriminator, self).__init__()
        
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        # Fusion layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(2 * protein_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Real/Fake classification only
        self.d_real_fake = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        molecular_token_features: torch.Tensor,
        protein_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Pool molecular features
        molecular_features = molecular_token_features.mean(dim=1)
        
        # Concatenate and fuse
        concat_features = torch.cat([protein_features, molecular_features], dim=1)
        concat_features = self.fusion_fc(concat_features)
        
        # Real/Fake classification only
        real_fake_score = self.d_real_fake(concat_features)
        
        return {
            'real_fake': real_fake_score
        }