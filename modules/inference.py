"""
Module 4: Batch GPU Inference & OOM Recovery
Streams results via st.progress(), catches CUDA OOM and halves batch size.
"""

from __future__ import annotations
import gc
import math
import time
from typing import List, Tuple, Optional

import torch
import torch.cuda.amp as amp
import streamlit as st

from modules.graph_pipeline import build_dataloader, LigandGraphDataset
from modules.model import DeepDockGNN
from rdkit import Chem


# ── Helpers ───────────────────────────────────────────────────────────────────
def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "node_feats": batch["node_feats"].to(device, non_blocking=True),
        "adj":        batch["adj"].to(device, non_blocking=True),
        "mask":       batch["mask"].to(device, non_blocking=True),
        "mols":       batch["mols"],
    }


def _run_one_batch(
    model:  DeepDockGNN,
    batch:  dict,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    """Forward pass with optional AMP autocast. Returns ΔG tensor on CPU."""
    b = _move_batch(batch, device)
    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with amp.autocast():
                scores = model(b["node_feats"], b["adj"], b["mask"])
        else:
            scores = model(b["node_feats"], b["adj"], b["mask"])
    return scores.float().cpu()


# ── Main inference entry point ────────────────────────────────────────────────
def run_inference(
    model:       DeepDockGNN,
    device:      torch.device,
    mols:        List[Chem.Mol],
    batch_size:  int = 32,
    use_fp16:    bool = True,
    progress_bar = None,          # st.progress() handle
    status_text  = None,          # st.empty() handle
) -> Tuple[List[Chem.Mol], List[float]]:
    """
    Run GNN inference over all mols with OOM-safe batching.

    Returns
    -------
    scored_mols : list of RDKit Mol objects
    scores      : list of ΔG values (kcal/mol)
    """
    if not mols:
        return [], []

    # Build initial dataloader
    loader, dataset = build_dataloader(mols, batch_size=batch_size)
    total_batches   = len(loader)

    all_scores: List[float] = []
    all_mols:   List[Chem.Mol] = []

    batch_idx = 0
    # Convert loader to list so we can restart on OOM
    batches = list(loader)

    while batch_idx < len(batches):
        batch = batches[batch_idx]

        try:
            scores = _run_one_batch(model, batch, device, use_fp16)
            all_scores.extend(scores.tolist())
            all_mols.extend(batch["mols"])
            batch_idx += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                # ── OOM Recovery: halve batch size ───────────────────────────
                torch.cuda.empty_cache()
                gc.collect()

                new_bs = max(1, batch_size // 2)
                if status_text:
                    status_text.warning(
                        f"CUDA OOM — reducing batch size {batch_size} → {new_bs}"
                    )
                batch_size = new_bs

                # Rebuild remaining batches with smaller batch size
                remaining_mols = [m for b in batches[batch_idx:] for m in b["mols"]]
                remaining_loader, _ = build_dataloader(remaining_mols, batch_size=batch_size)
                batches = batches[:batch_idx] + list(remaining_loader)
                # Retry same batch_idx with new smaller batches
                continue

            else:
                raise  # non-OOM error → propagate

        # ── Progress update ───────────────────────────────────────────────────
        frac = min(batch_idx / max(total_batches, 1), 1.0)
        if progress_bar:
            progress_bar.progress(frac)
        if status_text:
            n_done = len(all_mols)
            status_text.text(
                f"Scoring… {n_done}/{len(mols)} ligands  |  "
                f"batch_size={batch_size}  |  "
                f"device={device.type.upper()}"
            )

    if progress_bar:
        progress_bar.progress(1.0)

    return all_mols, all_scores


def rank_results(
    mols:   List[Chem.Mol],
    scores: List[float],
    top_n:  int = 100,
) -> Tuple[List[Chem.Mol], List[float]]:
    """Sort by ΔG (most negative = best binding) and return top N."""
    paired = sorted(zip(scores, mols), key=lambda x: x[0])
    top    = paired[:top_n]
    top_scores, top_mols = zip(*top) if top else ([], [])
    return list(top_mols), list(top_scores)
