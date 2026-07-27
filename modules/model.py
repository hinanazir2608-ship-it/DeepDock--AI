"""
Module 3: PyTorch GPU Docking Engine
GNN-based binding affinity predictor with CUDA + FP16 AMP.
Loaded once via @st.cache_resource.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import streamlit as st


# ── GCN Layer (pure PyTorch, no PyG) ─────────────────────────────────────────
class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer: H' = σ(Â · H · W)
    where Â is the pre-normalised adjacency (D^{-1/2} A D^{-1/2}).
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear  = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.norm    = nn.LayerNorm(out_features)

    def forward(
        self,
        h: torch.Tensor,    # [B, N, F_in]
        adj: torch.Tensor,  # [B, N, N]
    ) -> torch.Tensor:      # [B, N, F_out]
        # Message passing: aggregate neighbour features
        agg = torch.bmm(adj, h)           # [B, N, F_in]
        out = self.linear(agg)             # [B, N, F_out]
        out = self.norm(F.gelu(out))
        out = self.dropout(out)
        return out


# ── Global Attention Pooling ──────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """Soft-attention global pooling over node representations."""

    def __init__(self, in_features: int):
        super().__init__()
        self.gate = nn.Linear(in_features, 1)

    def forward(
        self,
        h: torch.Tensor,    # [B, N, F]
        mask: torch.Tensor, # [B, N]  bool
    ) -> torch.Tensor:      # [B, F]
        logits = self.gate(h).squeeze(-1)           # [B, N]
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)      # [B, N]
        pooled = (weights.unsqueeze(-1) * h).sum(1) # [B, F]
        return pooled


# ── Main GNN Docking Model ────────────────────────────────────────────────────
class DeepDockGNN(nn.Module):
    """
    Graph Neural Network for predicting binding free energy (ΔG, kcal/mol).

    Architecture
    ------------
    3 × GCN layers (128-dim) → Attention Pooling → MLP head → scalar ΔG
    """

    NODE_FEATURES = 40   # matches atom_features() in graph_pipeline.py
    HIDDEN_DIM    = 128
    MLP_HIDDEN    = 256

    def __init__(self):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.NODE_FEATURES, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.GELU(),
        )

        # GCN layers
        self.gcn1 = GCNLayer(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.gcn2 = GCNLayer(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.gcn3 = GCNLayer(self.HIDDEN_DIM, self.HIDDEN_DIM)

        # Skip connection projection (residual)
        self.skip = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)

        # Global pooling
        self.pool = AttentionPooling(self.HIDDEN_DIM)

        # MLP head → binding score
        self.mlp = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, self.MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.MLP_HIDDEN, self.MLP_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.MLP_HIDDEN // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_feats: torch.Tensor,  # [B, N, F]
        adj:        torch.Tensor,  # [B, N, N]
        mask:       torch.Tensor,  # [B, N]
    ) -> torch.Tensor:             # [B]  ΔG values
        h = self.input_proj(node_feats)   # [B, N, H]

        # GCN stack with residual
        h1 = self.gcn1(h, adj)
        h2 = self.gcn2(h1, adj)
        h3 = self.gcn3(h2, adj) + self.skip(h)   # skip from input

        # Global representation
        g = self.pool(h3, mask)    # [B, H]

        # Binding affinity (kcal/mol, negative = stronger binding)
        dg = self.mlp(g).squeeze(-1) * -10.0 - 5.0  # calibrate to ~[-15, -2]

        return dg


# ── Model loader (cached in Streamlit session) ────────────────────────────────
@st.cache_resource(show_spinner="Loading DeepDock-AI model onto GPU…")
def load_model(weights_path: str = "") -> Tuple[DeepDockGNN, torch.device, str]:
    """
    Load (or initialise) the GNN model, move to CUDA/CPU.
    Uses @st.cache_resource so it runs only once per Streamlit session.

    Parameters
    ----------
    weights_path : path to a .pt checkpoint; empty → random-init demo model

    Returns
    -------
    model  : DeepDockGNN (eval mode, on device)
    device : torch.device
    precision : 'fp16' | 'fp32'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DeepDockGNN()

    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        # Demo mode: model is randomly initialised
        pass

    model = model.to(device)
    model.eval()

    # Detect FP16 support
    precision = "fp16" if device.type == "cuda" else "fp32"

    return model, device, precision


def get_device_info() -> dict:
    """Return GPU/CPU info for display in the Streamlit sidebar."""
    info = {
        "device": "CPU",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_gb": None,
        "precision": "fp32",
    }
    if torch.cuda.is_available():
        info["device"]    = f"CUDA:{torch.cuda.current_device()}"
        info["gpu_name"]  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"]   = round(vram / 1e9, 1)
        info["precision"] = "fp16 (AMP)"
    return info
