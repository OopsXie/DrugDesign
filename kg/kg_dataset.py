"""
Knowledge Graph Dataset Module for Drug-MSM

Handles biomedical KG construction from TSV data, k-hop subgraph extraction,
and triple preparation for TransE training.

Entity types: Protein (UniProt), Drug (DrugBank), Disease (DOID), Pathway (KEGG)
Relation types: targets, participates_in, associated_with, treats
"""

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch.utils.data import Dataset


ENTITY_TYPES = ['Protein', 'Drug', 'Disease', 'Pathway']
RELATION_TYPES = ['targets', 'participates_in', 'associated_with', 'treats']


class KGDataset(Dataset):
    """
    Knowledge Graph Dataset with adjacency list storage.
    
    Stores KG as: {entity_id: {relation_type: [neighbor_ids]}}
    Provides entity/relation vocabularies and index mappings.
    """
    
    def __init__(self):
        self.adjacency: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.reverse_adjacency: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}
        
        self.entity_types: Dict[str, str] = {}
        self.triples: List[Tuple[str, str, str]] = []
        
    @property
    def num_entities(self) -> int:
        return len(self.entity_to_idx)
    
    @property
    def num_relations(self) -> int:
        return len(self.relation_to_idx)
    
    @property
    def entity_vocab(self) -> Dict[str, int]:
        return self.entity_to_idx.copy()
    
    @property
    def relation_vocab(self) -> Dict[str, int]:
        return self.relation_to_idx.copy()
    
    def _add_entity(self, entity_id: str, entity_type: str) -> int:
        if entity_id not in self.entity_to_idx:
            idx = len(self.entity_to_idx)
            self.entity_to_idx[entity_id] = idx
            self.idx_to_entity[idx] = entity_id
            self.entity_types[entity_id] = entity_type
        return self.entity_to_idx[entity_id]
    
    def _add_relation(self, relation: str) -> int:
        if relation not in self.relation_to_idx:
            idx = len(self.relation_to_idx)
            self.relation_to_idx[relation] = idx
            self.idx_to_relation[idx] = relation
        return self.relation_to_idx[relation]
    
    def add_triple(self, head: str, relation: str, tail: str,
                   head_type: Optional[str] = None, tail_type: Optional[str] = None):
        if head_type is None:
            head_type = self._infer_entity_type(head)
        if tail_type is None:
            tail_type = self._infer_entity_type(tail)
        
        self._add_entity(head, head_type)
        self._add_entity(tail, tail_type)
        rel_idx = self._add_relation(relation)
        
        self.adjacency[head][relation].append(tail)
        self.reverse_adjacency[tail][relation].append(head)
        self.triples.append((head, relation, tail))
    
    def _infer_entity_type(self, entity_id: str) -> str:
        if ':' in entity_id:
            prefix = entity_id.split(':')[0].upper()
            if prefix in ['UNIPROT', 'P']:
                return 'Protein'
            elif prefix in ['DRUGBANK', 'DB']:
                return 'Drug'
            elif prefix in ['DOID', 'DO']:
                return 'Disease'
            elif prefix in ['KEGG', 'PATHWAY', 'HS']:
                return 'Pathway'
        
        if entity_id.startswith('P') and entity_id[1:].isdigit():
            return 'Protein'
        elif entity_id.startswith('DB') and entity_id[2:].isdigit():
            return 'Drug'
        elif entity_id.startswith('DOID:'):
            return 'Disease'
        elif entity_id.startswith('hsa') or entity_id.startswith('PATHWAY'):
            return 'Pathway'
        
        return 'Protein'
    
    @classmethod
    def build_kg_from_tsv(cls, triples_path: str) -> 'KGDataset':
        """
        Build KG from TSV file.
        
        Supports two formats:
        1. Full format: head_entity, relation, tail_entity, head_type, tail_type
        2. Simple format: head, relation, tail (auto-type detection)
        
        Example full format:
            UniProt:P00533\tassociated_with\tDOID:3908\tProtein\tDisease
        
        Example simple format:
            P00533\ttargets\tDB00123
        """
        dataset = cls()
        path = Path(triples_path)
        
        if not path.exists():
            raise FileNotFoundError(f"KG TSV file not found: {triples_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                
                if len(parts) >= 5:
                    head, relation, tail, head_type, tail_type = parts[:5]
                elif len(parts) == 3:
                    head, relation, tail = parts
                    head_type = None
                    tail_type = None
                else:
                    continue
                
                dataset.add_triple(head, relation, tail, head_type, tail_type)
        
        return dataset
    
    def extract_khop_subgraph(self, target_protein: str, k: int = 2,
                               topk: Optional[int] = None) -> 'KGDataset':
        """
        Extract k-hop neighborhood around target protein using BFS.
        
        If topk is set, caps neighbors per relation type to topk (Formula 4-11).
        
        Args:
            target_protein: Target protein entity ID
            k: Number of hops
            topk: Maximum neighbors per relation type (optional)
        
        Returns:
            New KGDataset containing the subgraph
        """
        if target_protein not in self.entity_to_idx:
            raise ValueError(f"Target protein not in KG: {target_protein}")
        
        subgraph = KGDataset()
        visited: Set[str] = set()
        current_level: Set[str] = {target_protein}
        
        target_type = self.entity_types.get(target_protein, 'Protein')
        subgraph._add_entity(target_protein, target_type)
        visited.add(target_protein)
        
        for hop in range(k):
            next_level: Set[str] = set()
            
            for entity in current_level:
                if entity not in self.adjacency:
                    continue
                
                for relation, neighbors in self.adjacency[entity].items():
                    if topk is not None:
                        neighbors = random.sample(neighbors, min(topk, len(neighbors)))
                    
                    for neighbor in neighbors:
                        neighbor_type = self.entity_types.get(neighbor, 'Protein')
                        subgraph._add_entity(neighbor, neighbor_type)
                        subgraph._add_relation(relation)
                        
                        subgraph.adjacency[entity][relation].append(neighbor)
                        subgraph.reverse_adjacency[neighbor][relation].append(head)
                        subgraph.triples.append((entity, relation, neighbor))
                        
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_level.add(neighbor)
            
            current_level = next_level
            if not current_level:
                break
        
        return subgraph
    
    def get_train_triples(self) -> torch.Tensor:
        """
        Return tensor of shape (num_triples, 3) with [head_idx, rel_idx, tail_idx].
        """
        triples_list = []
        for head, relation, tail in self.triples:
            h_idx = self.entity_to_idx[head]
            r_idx = self.relation_to_idx[relation]
            t_idx = self.entity_to_idx[tail]
            triples_list.append([h_idx, r_idx, t_idx])
        
        return torch.tensor(triples_list, dtype=torch.long)
    
    def get_entity_type_mask(self) -> Dict[str, List[int]]:
        """
        Return mapping from entity type to list of entity indices.
        Used for type-constrained negative sampling.
        """
        type_mask: Dict[str, List[int]] = defaultdict(list)
        for entity_id, idx in self.entity_to_idx.items():
            entity_type = self.entity_types.get(entity_id, 'Protein')
            type_mask[entity_type].append(idx)
        return dict(type_mask)
    
    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[str]:
        """Get neighbors of an entity, optionally filtered by relation type."""
        if entity_id not in self.adjacency:
            return []
        
        if relation_type is None:
            neighbors = []
            for rel_neighbors in self.adjacency[entity_id].values():
                neighbors.extend(rel_neighbors)
            return neighbors
        
        return self.adjacency[entity_id].get(relation_type, [])
    
    def get_all_neighbors(self, entity_id: str) -> Dict[str, List[str]]:
        """Get all neighbors grouped by relation type."""
        if entity_id not in self.adjacency:
            return {}
        return dict(self.adjacency[entity_id])
    
    def save(self, path: str, format: str = 'pickle'):
        """Save KG to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            if not path.suffix:
                path = path.with_suffix('.pkl')
            with open(path, 'wb') as f:
                pickle.dump({
                    'adjacency': dict(self.adjacency),
                    'reverse_adjacency': dict(self.reverse_adjacency),
                    'entity_to_idx': self.entity_to_idx,
                    'idx_to_entity': self.idx_to_entity,
                    'relation_to_idx': self.relation_to_idx,
                    'idx_to_relation': self.idx_to_relation,
                    'entity_types': self.entity_types,
                    'triples': self.triples
                }, f)
        elif format == 'json':
            if not path.suffix:
                path = path.with_suffix('.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    'adjacency': {k: {r: v for r, v in dict(v2).items()} for k, v2 in dict(self.adjacency).items()},
                    'entity_to_idx': self.entity_to_idx,
                    'relation_to_idx': self.relation_to_idx,
                    'entity_types': self.entity_types,
                    'triples': self.triples
                }, f, indent=2)
    
    @classmethod
    def load(cls, path: str, format: str = 'pickle') -> 'KGDataset':
        """Load KG from file."""
        path = Path(path)
        
        if format == 'pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            dataset = cls()
            dataset.adjacency = defaultdict(lambda: defaultdict(list))
            for k, v in data['adjacency'].items():
                for r, neighbors in v.items():
                    dataset.adjacency[k][r] = neighbors
            
            dataset.reverse_adjacency = defaultdict(lambda: defaultdict(list))
            if 'reverse_adjacency' in data:
                for k, v in data['reverse_adjacency'].items():
                    for r, neighbors in v.items():
                        dataset.reverse_adjacency[k][r] = neighbors
            
            dataset.entity_to_idx = data['entity_to_idx']
            dataset.idx_to_entity = {int(k): v for k, v in data['idx_to_entity'].items()}
            dataset.relation_to_idx = data['relation_to_idx']
            dataset.idx_to_relation = {int(k): v for k, v in data['idx_to_relation'].items()}
            dataset.entity_types = data['entity_types']
            dataset.triples = data['triples']
            
            return dataset
        
        elif format == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dataset = cls()
            dataset.adjacency = defaultdict(lambda: defaultdict(list))
            for k, v in data['adjacency'].items():
                for r, neighbors in v.items():
                    dataset.adjacency[k][r] = neighbors
            
            dataset.entity_to_idx = data['entity_to_idx']
            dataset.idx_to_entity = {int(k): v for k, v in data['entity_to_idx'].items()}
            dataset.relation_to_idx = data['relation_to_idx']
            dataset.idx_to_relation = {int(k): v for k, v in data['relation_to_idx'].items()}
            dataset.entity_types = data.get('entity_types', {})
            dataset.triples = [tuple(t) for t in data.get('triples', [])]
            
            for head, rel, tail in dataset.triples:
                if head in dataset.entity_types:
                    dataset.reverse_adjacency[tail][rel].append(head)
            
            return dataset
        
        raise ValueError(f"Unknown format: {format}")
    
    def __len__(self) -> int:
        return len(self.triples)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        head, relation, tail = self.triples[idx]
        return (
            self.entity_to_idx[head],
            self.relation_to_idx[relation],
            self.entity_to_idx[tail]
        )
    
    def __repr__(self) -> str:
        return (f"KGDataset(num_entities={self.num_entities}, "
                f"num_relations={self.num_relations}, "
                f"num_triples={len(self.triples)})")


def collate_triples(batch):
    """Collate function for DataLoader with KGDataset."""
    heads, rels, tails = zip(*batch)
    return (
        torch.tensor(heads, dtype=torch.long),
        torch.tensor(rels, dtype=torch.long),
        torch.tensor(tails, dtype=torch.long)
    )