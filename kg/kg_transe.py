"""
TransE Embedding Model for Knowledge Graph

Implements TransE with margin ranking loss, negative sampling,
and neighborhood aggregation for knowledge representation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .kg_dataset import KGDataset, collate_triples


class TransEModel(nn.Module):
    """
    TransE model for knowledge graph embedding.
    
    Energy function: f(h,r,t) = ||h(h) + r(r) - h(t)||_2
    """
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def get_entity_embedding(self, entity_idx: torch.Tensor) -> torch.Tensor:
        return self.entity_embeddings(entity_idx)
    
    def get_relation_embedding(self, relation_idx: torch.Tensor) -> torch.Tensor:
        return self.relation_embeddings(relation_idx)
    
    def energy(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute TransE energy score.
        
        f(h,r,t) = ||h(h) + r(r) - h(t)||_2
        
        Args:
            h: Head entity embeddings (batch, embedding_dim)
            r: Relation embeddings (batch, embedding_dim)
            t: Tail entity embeddings (batch, embedding_dim)
        
        Returns:
            Energy scores (batch,)
        """
        return torch.norm(h + r - t, p=2, dim=1)
    
    def forward(self, pos_h: torch.Tensor, pos_r: torch.Tensor, pos_t: torch.Tensor,
                neg_h: torch.Tensor, neg_r: torch.Tensor, neg_t: torch.Tensor,
                margin: float = 1.0) -> torch.Tensor:
        """
        Compute margin ranking loss for TransE.
        
        L_TransE = sum [γ + f(h,r,t) - f(h',r',t')]_+
        
        Args:
            pos_h, pos_r, pos_t: Positive triple indices (batch,)
            neg_h, neg_r, neg_t: Negative triple indices (batch,)
            margin: Margin γ (default: 1.0)
        
        Returns:
            Scalar loss value
        """
        pos_h_emb = self.get_entity_embedding(pos_h)
        pos_r_emb = self.get_relation_embedding(pos_r)
        pos_t_emb = self.get_entity_embedding(pos_t)
        
        neg_h_emb = self.get_entity_embedding(neg_h)
        neg_r_emb = self.get_relation_embedding(neg_r)
        neg_t_emb = self.get_entity_embedding(neg_t)
        
        pos_energy = self.energy(pos_h_emb, pos_r_emb, pos_t_emb)
        neg_energy = self.energy(neg_h_emb, neg_r_emb, neg_t_emb)
        
        loss = F.relu(pos_energy - neg_energy + margin)
        return loss.mean()
    
    def constrain_entity_norm(self, threshold: float = 1.0):
        """
        Project entity embeddings to unit ball.
        
        ||h(e)||_2 <= threshold
        """
        with torch.no_grad():
            entity_norms = torch.norm(self.entity_embeddings.weight, p=2, dim=1, keepdim=True)
            scale = torch.clamp(entity_norms / threshold, min=1.0)
            self.entity_embeddings.weight.data.div_(scale)
    
    def get_all_entity_embeddings(self) -> torch.Tensor:
        return self.entity_embeddings.weight.data.clone()
    
    def get_all_relation_embeddings(self) -> torch.Tensor:
        return self.relation_embeddings.weight.data.clone()


def corrupt_triples(triples: torch.Tensor, num_entities: int,
                    entity_type_mask: Dict[str, List[int]],
                    entity_types: Dict[str, str],
                    n_neg: int = 1,
                    corruption_type: str = 'random') -> torch.Tensor:
    """
    Generate negative samples by corrupting triples.
    
    For each (h,r,t):
    - 50% chance: corrupt head (sample h' != h, same type)
    - 50% chance: corrupt tail (sample t' != t, same type)
    
    NOTE: This vectorized version uses random sampling without type constraints
    for performance. Type-constrained sampling can be added back if needed.
    
    Args:
        triples: Positive triples (batch, 3) with [h, r, t]
        num_entities: Total number of entities
        entity_type_mask: Mapping from entity type to list of entity indices
        entity_types: Mapping from entity_id to entity type
        n_neg: Number of negatives per positive
        corruption_type: 'random' or 'type_constrained' (currently both use random)
    
    Returns:
        Corrupted triples (batch * n_neg, 3)
    """
    batch_size = triples.shape[0]
    device = triples.device
    
    # Randomly decide which to corrupt: head or tail (50/50)
    corrupt_head = torch.rand(batch_size, device=device) < 0.5
    
    # Start with copies of original triples
    neg_h = triples[:, 0].clone()
    neg_r = triples[:, 1].clone()
    neg_t = triples[:, 2].clone()
    
    # Generate n_neg corrupted versions for each triple
    all_corrupted = []
    
    for neg_idx in range(n_neg):
        # Generate random entities for corruption
        new_h = torch.randint(0, num_entities, (batch_size,), device=device)
        new_t = torch.randint(0, num_entities, (batch_size,), device=device)
        
        # Avoid self-corruption: resample if new entity equals original
        # For head corruption
        head_collision = (new_h == triples[:, 0])
        while head_collision.any():
            new_h[head_collision] = torch.randint(0, num_entities, (head_collision.sum(),), device=device)
            head_collision = (new_h == triples[:, 0])
        
        # For tail corruption
        tail_collision = (new_t == triples[:, 2])
        while tail_collision.any():
            new_t[tail_collision] = torch.randint(0, num_entities, (tail_collision.sum(),), device=device)
            tail_collision = (new_t == triples[:, 2])
        
        # Apply corruption based on corrupt_head mask
        neg_h = torch.where(corrupt_head, new_h, triples[:, 0])
        neg_t = torch.where(~corrupt_head, new_t, triples[:, 2])
        
        # Stack corrupted triples
        corrupted_batch = torch.stack([neg_h, neg_r, neg_t], dim=1)
        all_corrupted.append(corrupted_batch)
    
    # Concatenate all negative samples
    corrupted = torch.cat(all_corrupted, dim=0)
    
    return corrupted


def train_transe(kg_dataset: KGDataset, embedding_dim: int = 100, epochs: int = 100,
                 batch_size: int = 512, lr: float = 0.01, margin: float = 1.0,
                 n_neg: int = 1, device: str = 'cpu',
                 save_path: Optional[str] = None,
                 verbose: bool = True) -> TransEModel:
    """
    Train TransE model on knowledge graph.
    
    Args:
        kg_dataset: KGDataset with triples
        embedding_dim: Embedding dimension (default: 100)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 512)
        lr: Learning rate (default: 0.01)
        margin: Margin for ranking loss (default: 1.0)
        n_neg: Number of negative samples per positive (default: 1)
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save trained embeddings (optional)
        verbose: Print training progress
    
    Returns:
        Trained TransEModel
    """
    model = TransEModel(
        num_entities=kg_dataset.num_entities,
        num_relations=kg_dataset.num_relations,
        embedding_dim=embedding_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    triples_tensor = kg_dataset.get_train_triples()
    entity_type_mask = kg_dataset.get_entity_type_mask()
    
    dataloader = DataLoader(
        kg_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_triples,
        num_workers=0
    )
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_h, batch_r, batch_t in dataloader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)
            
            pos_triples = torch.stack([batch_h, batch_r, batch_t], dim=1)
            
            neg_triples = corrupt_triples(
                pos_triples,
                kg_dataset.num_entities,
                entity_type_mask,
                kg_dataset.entity_types,
                n_neg=n_neg
            ).to(device)
            
            neg_h = neg_triples[:, 0]
            neg_r = neg_triples[:, 1]
            neg_t = neg_triples[:, 2]
            
            optimizer.zero_grad()
            
            loss = model(batch_h, batch_r, batch_t, neg_h, neg_r, neg_t, margin=margin)
            
            loss.backward()
            optimizer.step()
            
            model.constrain_entity_norm(threshold=1.0)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_entities': kg_dataset.num_entities,
            'num_relations': kg_dataset.num_relations,
            'embedding_dim': embedding_dim,
            'entity_to_idx': kg_dataset.entity_to_idx,
            'relation_to_idx': kg_dataset.relation_to_idx
        }, save_path)
        if verbose:
            print(f"Model saved to {save_path}")
    
    return model


def aggregate_neighborhood(model: TransEModel, kg_dataset: KGDataset,
                            target_protein: str, k: int = 1) -> torch.Tensor:
    """
    Aggregate neighborhood embeddings for knowledge representation.
    
    h_kg = (1/|S_p|) * Σ_{v∈S_p} h(v)
    
    where S_p = N_1(e_p) ∪ {e_p}
    
    Args:
        model: Trained TransEModel
        kg_dataset: KGDataset
        target_protein: Target protein entity ID
        k: Number of hops (currently supports k=1)
    
    Returns:
        Aggregated knowledge embedding (embedding_dim,)
    """
    if target_protein not in kg_dataset.entity_to_idx:
        raise ValueError(f"Target protein not in KG: {target_protein}")
    
    target_idx = kg_dataset.entity_to_idx[target_protein]
    target_tensor = torch.tensor([target_idx], dtype=torch.long)
    target_emb = model.get_entity_embedding(target_tensor).squeeze(0)
    
    neighbors = kg_dataset.get_all_neighbors(target_protein)
    neighbor_indices = []
    
    for relation, neighbor_ids in neighbors.items():
        for neighbor_id in neighbor_ids:
            if neighbor_id in kg_dataset.entity_to_idx:
                neighbor_indices.append(kg_dataset.entity_to_idx[neighbor_id])
    
    if not neighbor_indices:
        return target_emb
    
    neighbor_tensor = torch.tensor(neighbor_indices, dtype=torch.long)
    neighbor_embs = model.get_entity_embedding(neighbor_tensor)
    
    all_embs = torch.cat([target_emb.unsqueeze(0), neighbor_embs], dim=0)
    aggregated = all_embs.mean(dim=0)
    
    return aggregated


class KnowledgeProjector(nn.Module):
    """
    Project KG embedding to same dimension as sequence embedding.
    
    h̃_kg = W_p * h_kg + b_p
    
    Formula 4-15
    """
    
    def __init__(self, kg_dim: int, target_dim: int):
        super().__init__()
        self.kg_dim = kg_dim
        self.target_dim = target_dim
        self.projection = nn.Linear(kg_dim, target_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, h_kg: torch.Tensor) -> torch.Tensor:
        """
        Project KG embedding to target dimension.
        
        Args:
            h_kg: KG embedding (..., kg_dim)
        
        Returns:
            Projected embedding (..., target_dim)
        """
        return self.projection(h_kg)


def get_knowledge_representation(model: TransEModel, kg_dataset: KGDataset,
                                  projector: KnowledgeProjector,
                                  target_protein: str, k: int = 1) -> torch.Tensor:
    """
    End-to-end knowledge representation pipeline.
    
    1. Extract k-hop neighborhood of target protein
    2. Aggregate neighbor embeddings
    3. Project to target dimension
    
    Args:
        model: Trained TransEModel
        kg_dataset: KGDataset
        projector: KnowledgeProjector to align dimensions
        target_protein: Target protein entity ID
        k: Number of hops for neighborhood
    
    Returns:
        Knowledge representation h̃_kg (target_dim,)
    """
    h_kg = aggregate_neighborhood(model, kg_dataset, target_protein, k=k)
    h_kg_projected = projector(h_kg.unsqueeze(0)).squeeze(0)
    return h_kg_projected


def load_transe_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[TransEModel, Dict]:
    """
    Load trained TransE model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        device: Device to load model on
    
    Returns:
        Tuple of (model, metadata_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = TransEModel(
        num_entities=checkpoint['num_entities'],
        num_relations=checkpoint['num_relations'],
        embedding_dim=checkpoint['embedding_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = {
        'entity_to_idx': checkpoint.get('entity_to_idx', {}),
        'relation_to_idx': checkpoint.get('relation_to_idx', {})
    }
    
    return model, metadata