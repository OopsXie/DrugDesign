"""
KG-DrugGAN-MSM: Knowledge Graph Enhanced Molecular Generation Model

Chapter 4 of the thesis. Extends DrugGAN-MSM with:
- Biomedical knowledge graph embeddings (TransE)
- Gated fusion mechanism for dynamic sequence+knowledge integration
- Knowledge-enhanced conditional generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import os


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for dynamic integration of sequence and knowledge features.
    
    Formula 4-16: g = σ(W_seq·h_seq + W_kg·h̃_kg + b_g)
    Formula 4-17: h_fuse = g ⊙ h_seq + (1-g) ⊙ h̃_kg
    
    Args:
        d_model: Dimension of input features (must be same for both inputs)
    """
    
    def __init__(self, d_model: int):
        super(GatedFusion, self).__init__()
        # Gate computation (Formula 4-16)
        self.W_seq = nn.Linear(d_model, d_model)
        self.W_kg = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, h_seq: torch.Tensor, h_kg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_seq: Sequence features, shape (batch, d_model)
            h_kg: Knowledge features, shape (batch, d_model)
        Returns:
            h_fuse: Fused features (batch, d_model)
            gate_values: Gate values for inspection (batch, d_model)
        """
        # Formula 4-16: g = σ(W_seq·h_seq + W_kg·h_kg + b_g)
        g = self.sigmoid(self.W_seq(h_seq) + self.W_kg(h_kg))
        
        # Formula 4-17: h_fuse = g ⊙ h_seq + (1-g) ⊙ h_kg
        h_fuse = g * h_seq + (1 - g) * h_kg
        
        return h_fuse, g


class KnowledgeProjector(nn.Module):
    """
    Linear projection to align KG embedding dimension with sequence dimension.
    
    Formula 4-15: h̃_kg = W_p · h_kg + b_p
    
    Args:
        kg_dim: Input dimension from TransE embeddings
        target_dim: Target dimension (same as d_model, default 512)
    """
    
    def __init__(self, kg_dim: int = 100, target_dim: int = 512):
        super(KnowledgeProjector, self).__init__()
        self.projection = nn.Linear(kg_dim, target_dim)
    
    def forward(self, h_kg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_kg: Knowledge embedding (batch, kg_dim) or (kg_dim,)
        Returns:
            Projected knowledge embedding (batch, target_dim) or (target_dim,)
        """
        return self.projection(h_kg)


class KGConditionBuilder(nn.Module):
    """
    End-to-end knowledge condition builder.
    
    Combines:
    1. TransE embeddings (loaded as fixed/non-trainable from pre-trained TransE)
    2. Neighborhood aggregation
    3. Linear projection (trainable)
    4. Gated fusion (trainable)
    
    Args:
        entity_embeddings: Pre-computed TransE entity embeddings (num_entities, kg_dim)
        kg_adjacency: KG adjacency info for neighborhood aggregation
        d_model: Target dimension (512)
        kg_dim: TransE embedding dimension (100)
    """
    
    def __init__(self, entity_embeddings: torch.Tensor, kg_adjacency: Dict,
                 d_model: int = 512, kg_dim: int = 100):
        super(KGConditionBuilder, self).__init__()
        
        # Store TransE embeddings as buffer (non-trainable)
        self.register_buffer('entity_embeddings', entity_embeddings)
        
        # Build neighborhood aggregation mapping
        # kg_adjacency should be a dict: {entity_idx: [neighbor_indices]}
        self.kg_adjacency = kg_adjacency
        self.kg_dim = kg_dim
        self.d_model = d_model
        
        # Knowledge projector (Formula 4-15)
        self.projector = KnowledgeProjector(kg_dim, d_model)
        
        # Gated fusion (Formulas 4-16, 4-17)
        self.fusion = GatedFusion(d_model)
    
    def get_kg_embedding_for_protein(self, protein_entity_idx: int) -> torch.Tensor:
        """
        Get aggregated KG embedding for a target protein:
        1. Find 1-hop neighbors
        2. Mean aggregate: h_kg = mean(h(e_p) ∪ {h(neighbor) for neighbor in neighbors})
        
        Args:
            protein_entity_idx: Protein entity index in KG
        Returns:
            (kg_dim,) aggregated embedding
        """
        device = self.entity_embeddings.device
        
        # Get self embedding
        self_embedding = self.entity_embeddings[protein_entity_idx]  # (kg_dim,)
        
        # Get neighbors from adjacency
        neighbors = self.kg_adjacency.get(protein_entity_idx, [])
        
        if len(neighbors) == 0:
            # No neighbors, return self embedding
            return self_embedding
        
        # Get neighbor embeddings
        neighbor_indices = torch.tensor(neighbors, dtype=torch.long, device=device)
        neighbor_embeddings = self.entity_embeddings[neighbor_indices]  # (num_neighbors, kg_dim)
        
        # Mean aggregate: h_kg = mean(h(e_p) ∪ {h(neighbor)})
        all_embeddings = torch.cat([self_embedding.unsqueeze(0), neighbor_embeddings], dim=0)
        aggregated = all_embeddings.mean(dim=0)  # (kg_dim,)
        
        return aggregated
    
    def forward(self, h_seq: torch.Tensor, protein_entity_indices: torch.Tensor) -> torch.Tensor:
        """
        Build knowledge-enhanced condition for generator/discriminator.
        
        Args:
            h_seq: Sequence features from ProteinEncoder (batch, d_model)
            protein_entity_indices: Protein entity indices in KG (batch,)
        Returns:
            h_fuse: Fused condition (batch, d_model)
        """
        batch_size = protein_entity_indices.shape[0]
        device = protein_entity_indices.device
        
        # Get KG embeddings for each protein in batch
        kg_embeddings = []
        for i in range(batch_size):
            protein_idx = protein_entity_indices[i].item()
            kg_emb = self.get_kg_embedding_for_protein(protein_idx)
            kg_embeddings.append(kg_emb)
        
        # Stack to (batch, kg_dim)
        h_kg = torch.stack(kg_embeddings, dim=0)  # (batch, kg_dim)
        
        # Project to d_model (Formula 4-15)
        h_kg_projected = self.projector(h_kg)  # (batch, d_model)
        
        # Gated fusion (Formulas 4-16, 4-17)
        h_fuse, gate_values = self.fusion(h_seq, h_kg_projected)
        
        return h_fuse


class KnowledgeConsistencyLoss(nn.Module):
    """
    Knowledge consistency constraint (Formula 4-22).
    
    L_kg = E_z[||h_mol - h̃_kg||_2^2]
    
    Where:
    - h_mol: Generated molecule representation (from generator or discriminator intermediate features)
    - h̃_kg: Aligned knowledge representation
    
    This constraint encourages generated molecules to align with the target protein's
    knowledge context in semantic space.
    """
    
    def __init__(self):
        super(KnowledgeConsistencyLoss, self).__init__()
    
    def forward(self, h_mol: torch.Tensor, h_kg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_mol: Molecule features (batch, d_model)
            h_kg: Knowledge features (batch, d_model)
        Returns:
            MSE loss (scalar)
        """
        return F.mse_loss(h_mol, h_kg)


class KGDrugGANModel(nn.Module):
    """
    Complete KG-DrugGAN-MSM model.
    
    Wraps the components from DrugGAN_MSM and adds KG enhancement:
    - Uses ProteinEncoder from DrugGAN_MSM for sequence features
    - Uses KGConditionBuilder for knowledge enhancement
    - Uses Generator from DrugGAN_MSM with fused condition (instead of just h_seq)
    - Uses MultiTaskDiscriminator from DrugGAN_MSM with fused condition
    
    Args:
        pro_voc_len: Protein vocabulary size
        smi_voc_len: SMILES vocabulary size
        d_model: Model dimension (512)
        kg_dim: KG embedding dimension (100)
        entity_embeddings: Pre-trained TransE entity embeddings
        kg_adjacency: KG adjacency information
        mask_rate: MLM masking rate (0.15)
        noise_dim: Generator noise dimension (128)
        num_decoder_layers: Number of decoder layers in generator (12, matching paper)
    """
    
    def __init__(self, pro_voc_len: int, smi_voc_len: int, 
                 d_model: int = 756, kg_dim: int = 100,
                 entity_embeddings: Optional[torch.Tensor] = None,
                 kg_adjacency: Optional[Dict] = None,
                 mask_rate: float = 0.15,
                 noise_dim: int = 128,
                 num_decoder_layers: int = 12):
        super(KGDrugGANModel, self).__init__()
        
        # Import from DrugGAN_MSM (Chapter 3 components)
        from model.DrugGAN_MSM import ProteinEncoder, SMILESEncoder, Generator, MultiTaskDiscriminator, PositionalEncoding
        
        self.d_model = d_model
        self.kg_dim = kg_dim
        self.noise_dim = noise_dim
        
        # Sequence encoding modules (from Chapter 3)
        # d_model=756 to match ProteinEncoder/SMILESEncoder output_proj dimensions
        self.protein_encoder = ProteinEncoder(pro_voc_len, d_model, mask_rate)
        self.smi_encoder = SMILESEncoder(smi_voc_len, d_model)
        
        # KG enhancement
        self.use_kg = False
        if entity_embeddings is not None and kg_adjacency is not None:
            self.kg_condition_builder = KGConditionBuilder(
                entity_embeddings, kg_adjacency, d_model, kg_dim
            )
            self.use_kg = True
        
        # Generator and Discriminator (reuse from DrugGAN_MSM)
        # protein_dim=756 to match encoder output dimensions
        self.generator = Generator(smi_voc_len, d_model, noise_dim, num_layers=num_decoder_layers, protein_dim=756)
        self.discriminator = MultiTaskDiscriminator(d_model, hidden_dim=d_model // 2, protein_dim=756)
        
        # Knowledge consistency loss
        self.kg_consistency_loss = KnowledgeConsistencyLoss()
    
    def forward(self, protein_tokens: torch.Tensor, smiles_tokens: torch.Tensor,
                protein_entity_indices: Optional[torch.Tensor] = None,
                z: Optional[torch.Tensor] = None, mode: str = 'generate') -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            protein_tokens: Protein sequence tokens (batch, pro_seq_len)
            smiles_tokens: SMILES tokens (batch, smi_seq_len) — for teacher-forced training
            protein_entity_indices: Protein entity ID in KG (batch,) — None means no KG
            z: Noise vector (batch, noise_dim)
            mode: 'generate' | 'encode' | 'discriminate'
        Returns:
            Dictionary with results based on mode
        """
        # Encode protein sequence
        protein_output = self.protein_encoder(protein_tokens, apply_mlm=(mode != 'generate'))
        h_seq = protein_output['protein_features']  # (batch, d_model)
        
        # Build knowledge-enhanced condition
        if self.use_kg and protein_entity_indices is not None:
            h_condition = self.kg_condition_builder(h_seq, protein_entity_indices)
            
            # Get projected KG embedding for consistency loss
            batch_size = protein_entity_indices.shape[0]
            kg_embeddings = []
            for i in range(batch_size):
                protein_idx = protein_entity_indices[i].item()
                kg_emb = self.kg_condition_builder.get_kg_embedding_for_protein(protein_idx)
                kg_embeddings.append(kg_emb)
            h_kg_raw = torch.stack(kg_embeddings, dim=0)  # (batch, kg_dim)
            h_kg_projected = self.kg_condition_builder.projector(h_kg_raw)  # (batch, d_model)
        else:
            h_condition = h_seq
            h_kg_projected = None
        
        results = {}
        
        if mode == 'generate':
            # Generate SMILES logits
            batch_size = protein_tokens.shape[0]
            seq_len = smiles_tokens.shape[1]
            
            # Handle noise vector
            if z is None:
                z = torch.zeros(batch_size, self.noise_dim, device=protein_tokens.device)
            
            # Generate with knowledge-enhanced condition
            smiles_logits = self.generator(smiles_tokens, h_condition, z)
            
            results['smiles_logits'] = smiles_logits
            results['protein_features'] = h_seq
            results['condition_features'] = h_condition
            
            # Add KG consistency info if available
            if h_kg_projected is not None:
                results['kg_features'] = h_kg_projected
        
        elif mode == 'encode':
            # Encode SMILES
            smi_output = self.smi_encoder(smiles_tokens)
            h_mol = smi_output['molecular_features']
            
            results['molecular_features'] = h_mol
            results['protein_features'] = h_seq
            results['condition_features'] = h_condition
            results['smi_token_features'] = smi_output['token_features']
            results['smi_token_logits'] = smi_output['token_logits']
            
            # Add MLM information
            results['mlm_logits'] = protein_output['mlm_logits']
            if 'mask' in protein_output:
                results['mask'] = protein_output['mask']
            
            # Add KG consistency info if available
            if h_kg_projected is not None:
                results['kg_features'] = h_kg_projected
        
        elif mode == 'discriminate':
            # Encode SMILES and run discriminator
            smi_output = self.smi_encoder(smiles_tokens)
            h_mol = smi_output['molecular_features']
            smi_token_features = smi_output['token_features']
            
            # Discriminate with knowledge-enhanced condition
            disc_output = self.discriminator(smi_token_features, h_condition)
            
            results.update(disc_output)
            results['molecular_features'] = h_mol
            results['protein_features'] = h_seq
            results['condition_features'] = h_condition
            
            # Add KG consistency info if available
            if h_kg_projected is not None:
                results['kg_features'] = h_kg_projected
        
        return results
    
    def compute_kg_consistency_loss(self, h_mol: torch.Tensor, h_kg: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge consistency loss (Formula 4-22).
        
        Args:
            h_mol: Molecule features (batch, d_model)
            h_kg: Knowledge features (batch, d_model)
        Returns:
            L_kg scalar
        """
        return self.kg_consistency_loss(h_mol, h_kg)


def load_pretrained_kg_embeddings(kg_path: str) -> Tuple[torch.Tensor, Dict]:
    """
    Load pre-trained TransE embeddings from .pt file.
    
    Args:
        kg_path: Path to .pt file containing entity embeddings and adjacency
    
    Returns:
        Tuple of (entity_embeddings, adjacency_info)
            - entity_embeddings: (num_entities, kg_dim) tensor
            - adjacency_info: dict mapping entity_idx -> [neighbor_indices]
    """
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"KG embeddings not found at {kg_path}")
    
    data = torch.load(kg_path, map_location='cpu')
    
    # Expected format: {'entity_embeddings': tensor, 'adjacency': dict}
    entity_embeddings = data.get('entity_embeddings', data.get('embeddings'))
    adjacency_info = data.get('adjacency', data.get('neighbors', {}))
    
    if entity_embeddings is None:
        raise ValueError("KG file must contain 'entity_embeddings' or 'embeddings' key")
    
    return entity_embeddings, adjacency_info


def kg_inference(model: KGDrugGANModel, protein_tokens: torch.Tensor, 
                 protein_entity_idx: Optional[int] = None,
                 num_samples: int = 10, max_len: int = 80,
                 smi_start_token: int = 1, smi_end_token: int = 2,
                 temperature: float = 1.0) -> torch.Tensor:
    """
    Generate molecules using KG-enhanced model with autoregressive decoding.
    
    Args:
        model: KGDrugGANModel instance
        protein_tokens: Protein sequence tokens (batch, pro_seq_len) or (pro_seq_len,)
        protein_entity_idx: Protein entity ID in KG (scalar) or None
        num_samples: Number of molecules to generate
        max_len: Maximum SMILES sequence length
        smi_start_token: Start token ID for SMILES
        smi_end_token: End token ID for SMILES
        temperature: Sampling temperature (1.0 = greedy, >1.0 = more diverse)
    
    Returns:
        Generated SMILES token sequences (num_samples, generated_len)
    """
    device = next(model.parameters()).device
    
    # Ensure protein_tokens is batched
    if protein_tokens.dim() == 1:
        protein_tokens = protein_tokens.unsqueeze(0)
    
    batch_size = protein_tokens.shape[0]
    
    # Expand to num_samples
    protein_tokens = protein_tokens.repeat(num_samples, 1)
    
    # Prepare entity indices if provided
    protein_entity_indices = None
    if protein_entity_idx is not None:
        protein_entity_indices = torch.full((num_samples,), protein_entity_idx, 
                                           dtype=torch.long, device=device)
    
    # Initialize with start token
    generated = torch.full((num_samples, 1), smi_start_token, dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        # Encode protein once
        protein_output = model.protein_encoder(protein_tokens, apply_mlm=False)
        h_seq = protein_output['protein_features']
        
        # Build KG condition if available
        if model.use_kg and protein_entity_indices is not None:
            h_condition = model.kg_condition_builder(h_seq, protein_entity_indices)
        else:
            h_condition = h_seq
        
        # Generate noise
        z = torch.randn(num_samples, model.noise_dim, device=device)
        
        # Autoregressive generation
        for step in range(max_len - 1):
            # Get logits for next token
            logits = model.generator(generated, h_condition, z)
            
            # Get logits for next position (last position)
            next_token_logits = logits[:, -1, :] / temperature  # (num_samples, smi_voc_len)
            
            # Sample or take argmax
            if temperature == 1.0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)  # (num_samples,)
            else:
                # Temperature-scaled sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
            
            # Check for end tokens
            if (next_token == smi_end_token).all():
                break
    
    return generated


class KGDrugGANTrainer:
    """
    Training utility for KG-DrugGAN-MSM model.
    
    Implements the combined loss from Formula 4-23:
    L_total = L_adv + αL_prop + βL_kg
    
    Where:
    - L_adv: Adversarial loss (GAN loss)
    - L_prop: Property prediction loss
    - L_kg: Knowledge consistency loss
    - α, β: Balancing hyperparameters
    """
    
    def __init__(self, model: KGDrugGANModel, lr_g: float = 1e-4, lr_d: float = 1e-4,
                 alpha: float = 0.1, beta: float = 0.01):
        """
        Args:
            model: KGDrugGANModel to train
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            alpha: Weight for property loss (α in Formula 4-23)
            beta: Weight for knowledge consistency loss (β in Formula 4-23)
        """
        self.model = model
        self.alpha = alpha
        self.beta = beta
        
        # Separate optimizers for G and D
        self.optimizer_g = torch.optim.Adam(
            list(model.generator.parameters()) + 
            list(model.protein_encoder.parameters()) +
            (list(model.kg_condition_builder.parameters()) if model.use_kg else []),
            lr=lr_g
        )
        
        self.optimizer_d = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=lr_d
        )
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kg_loss = KnowledgeConsistencyLoss()
    
    def train_step(self, protein_tokens: torch.Tensor, smiles_tokens: torch.Tensor,
                   protein_entity_indices: Optional[torch.Tensor] = None,
                   real_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            protein_tokens: Protein tokens (batch, pro_seq_len)
            smiles_tokens: Real SMILES tokens (batch, smi_seq_len)
            protein_entity_indices: Protein entity IDs (batch,)
            real_labels: Real/fake labels (batch, 1), defaults to ones
        
        Returns:
            Dictionary of loss components
        """
        batch_size = protein_tokens.shape[0]
        device = protein_tokens.device
        
        if real_labels is None:
            real_labels = torch.ones(batch_size, 1, device=device)
        
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # ========== Train Discriminator ==========
        self.optimizer_d.zero_grad()
        
        # Real samples
        real_output = self.model(
            protein_tokens, smiles_tokens, 
            protein_entity_indices=protein_entity_indices,
            mode='discriminate'
        )
        
        # Real loss
        d_real_loss = self.bce_loss(real_output['real_fake'], real_labels)
        
        # Property losses on real samples
        # Assume we have target properties (would be passed in real scenario)
        # For now, skip property loss computation
        
        # Generate fake samples
        z = torch.randn(batch_size, self.model.noise_dim, device=device)
        with torch.no_grad():
            gen_output = self.model(
                protein_tokens, smiles_tokens,
                protein_entity_indices=protein_entity_indices,
                z=z, mode='generate'
            )
        
        # Get generated SMILES (argmax)
        fake_smiles_logits = gen_output['smiles_logits']
        fake_smiles = torch.argmax(fake_smiles_logits, dim=-1)
        
        # Discriminate fake samples
        fake_output = self.model(
            protein_tokens, fake_smiles,
            protein_entity_indices=protein_entity_indices,
            mode='discriminate'
        )
        
        # Fake loss
        d_fake_loss = self.bce_loss(fake_output['real_fake'], fake_labels)
        
        # Total D loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.optimizer_d.step()
        
        # ========== Train Generator ==========
        self.optimizer_g.zero_grad()
        
        # Generate and discriminate
        gen_output = self.model(
            protein_tokens, smiles_tokens,
            protein_entity_indices=protein_entity_indices,
            z=z, mode='generate'
        )
        
        # Get generated SMILES
        fake_smiles_logits = gen_output['smiles_logits']
        fake_smiles = torch.argmax(fake_smiles_logits, dim=-1)
        
        # Discriminate to get G loss (want to fool D)
        fake_disc_output = self.model(
            protein_tokens, fake_smiles,
            protein_entity_indices=protein_entity_indices,
            mode='discriminate'
        )
        
        # Adversarial loss (want D to say real)
        g_adv_loss = self.bce_loss(fake_disc_output['real_fake'], real_labels)
        
        # Knowledge consistency loss (Formula 4-22): L_kg = ||h_mol - h_kg||²
        g_kg_loss = torch.tensor(0.0, device=device)
        if self.model.use_kg and protein_entity_indices is not None:
            # h_mol from fake_disc_output (encoded fake SMILES), h_kg from gen_output
            if 'kg_features' in gen_output and 'molecular_features' in fake_disc_output:
                h_mol = fake_disc_output['molecular_features']  # Generated molecule representation
                h_kg = gen_output['kg_features']  # Projected KG embedding
                g_kg_loss = self.kg_loss(h_mol, h_kg)
        
        # Total G loss (Formula 4-23)
        g_loss = g_adv_loss + self.beta * g_kg_loss
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item(),
            'g_loss': g_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_kg_loss': g_kg_loss.item()
        }