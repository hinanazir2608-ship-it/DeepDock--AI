"""
Module 2: PyG-Style Graph Pipeline (pure PyTorch, no PyG dependency)
Converts RDKit Mol objects into graph tensors for GNN inference.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem


# ── Atom feature encoding ───────────────────────────────────────────────
ATOM_TYPES = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I",
    "B", "Si", "Se", "Te", "other",
]

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices: list) -> List[int]:
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1
    return enc


def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """75-dimensional atom feature vector."""
    feats: List[float] = []
    feats += _one_hot(atom.GetSymbol(), ATOM_TYPES)                    # 15
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)    # 6
    feats.append(float(atom.GetIsAromatic()))                           # 1
    feats.append(float(atom.IsInRing()))                                # 1
    feats.append(float(atom.GetTotalNumHs()))                           # 1
    feats.append(float(atom.GetFormalCharge()))                         # 1
    feats.append(float(atom.GetDegree()))                               # 1
    feats.append(float(atom.GetImplicitValence()))                      # 1
    feats.append(float(atom.GetNumRadicalElectrons()))                  # 1
    feats.append(float(atom.GetMass()) / 100.0)                        # 1
    # Atomic number bucketed 0-118 → one-hot over 9 bins
    an = atom.GetAtomicNum()
    bins = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    feats += _one_hot(an, bins)                                         # 11
    return feats  # total 40


def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    """6-dimensional bond feature vector."""
    feats: List[float] = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)   # 5
    feats.append(float(bond.GetIsConjugated()))          # 1
    feats.append(float(bond.IsInRing()))                 # 1
    return feats  # total 7


def mol_to_graph(mol: Chem.Mol) -> Optional[dict]:
    """
    Convert an RDKit Mol to a graph dict with:
      - node_feats : FloatTensor [N, F_node]
      - adj        : FloatTensor [N, N]   (symmetric, with self-loops)
      - edge_feats : FloatTensor [N, N, F_edge]  (sparse via adj mask)
    Returns None if molecule is invalid or has < 2 atoms.
    """
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    if n < 2:
        return None

    # ── Node features ───────────────────────────────────────────────────────
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    node_dim = len(node_feats[0])
    node_tensor = torch.zeros(n, node_dim, dtype=torch.float32)
    for i, f in enumerate(node_feats):
        node_tensor[i] = torch.tensor(f, dtype=torch.float32)

    # ── Adjacency + edge features ───────────────────────────────────────────
    edge_dim = len(bond_features(mol.GetBondWithIdx(0))
                   ) if mol.GetNumBonds() > 0 else 7
    adj = torch.zeros(n, n, dtype=torch.float32)
    edge_feat_mat = torch.zeros(n, n, edge_dim, dtype=torch.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = torch.tensor(bond_features(bond), dtype=torch.float32)
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        edge_feat_mat[i, j] = bf
        edge_feat_mat[j, i] = bf

    # Self-loops
    adj += torch.eye(n, dtype=torch.float32)
    # Degree-normalise: D^{-1/2} A D^{-1/2}
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    return {
        "node_feats": node_tensor,   # [N, F_node]
        "adj": norm_adj,      # [N, N]
        "n_atoms": n,
        "mol": Chem.RemoveHs(mol),
    }


# ── Dataset ─────────────────────────────────────────────────────────────
class LigandGraphDataset(Dataset):
    """Wraps a list of RDKit Mol objects as graph-tensor items."""

    def __init__(self, mols: List[Chem.Mol]):
        self.graphs: List[dict] = []
        self.valid_mols: List[Chem.Mol] = []

        for mol in mols:
            g = mol_to_graph(mol)
            if g is not None:
                self.graphs.append(g)
                self.valid_mols.append(g["mol"])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]


def collate_graphs(batch: List[dict]) -> dict:
    """
    Pad and batch a list of variable-size graphs for GPU inference.
    Pads node_feats and adj to the max graph size in the batch.
    """
    max_n = max(g["n_atoms"] for g in batch)
    node_dim = batch[0]["node_feats"].shape[1]

    node_padded = torch.zeros(len(batch), max_n, node_dim)
    adj_padded = torch.zeros(len(batch), max_n, max_n)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)

    for i, g in enumerate(batch):
        n = g["n_atoms"]
        node_padded[i, :n, :] = g["node_feats"]
        adj_padded[i, :n, :n] = g["adj"]
        mask[i, :n] = True

    return {
        "node_feats": node_padded,
        "adj": adj_padded,
        "mask": mask,
        "mols": [g["mol"] for g in batch],
    }


def build_dataloader(
    mols: List[Chem.Mol],
    batch_size: int = 32,
) -> Tuple[DataLoader, LigandGraphDataset]:
    dataset = LigandGraphDataset(mols)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,   # must be 0 when called from Streamlit
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
