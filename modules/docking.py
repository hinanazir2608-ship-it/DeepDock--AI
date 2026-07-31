"""
Module: GNINA/AutoDock-Vina Docking Engine (DEBUG VERSION)
Uses AutoDock-Vina Python bindings (same backend as GNINA) for pose sampling.
GNINA-style CNN rescoring model included for GPU-accelerated affinity prediction.
"""

from __future__ import annotations
import os
import io
import gc
import time
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

try:
    from vina import Vina

    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

try:
    from meeko import MoleculePreparation, PDBQTMolecule, RDKitMolCreate

    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False


# ── PDBQT to PDB Conversion ───────────────────────────────────────────────────
def pdbqt_to_pdb_standard(pdbqt_content: str) -> str:
    """
    Convert PDBQT format (with partial charges) to standard PDB format.
    Keeps only ATOM/HETATM lines and strips PDBQT-specific columns.
    """
    pdb_lines = []
    for line in pdbqt_content.splitlines():
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            # Keep first 66 characters (standard PDB format)
            pdb_lines.append(line[:66])
    return "\n".join(pdb_lines) + "\n"


class GNINAConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GNINAScorer(nn.Module):
    IN_CHANNELS = 35
    GRID_SIZE = 24

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            GNINAConvBlock(self.IN_CHANNELS, 32, pool=False),
            GNINAConvBlock(32, 64, pool=True),
            GNINAConvBlock(64, 128, pool=True),
            GNINAConvBlock(128, 256, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.trunk = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.pose_head = nn.Linear(256, 1)
        self.affinity_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(grid)
        x = self.gap(x).flatten(1)
        x = self.trunk(x)
        pose = torch.sigmoid(self.pose_head(x)).squeeze(-1)
        affin = (self.affinity_head(x).squeeze(-1) * -10.0) - 5.0
        return pose, affin


@st.cache_resource(show_spinner="Loading GNINA scoring model…")
def load_gnina_model(weights_path: str = "") -> Tuple[GNINAScorer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNINAScorer().to(device).eval()
    if weights_path and Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    return model, device


def _mol_to_pdbqt_str(mol: Chem.Mol, name: str = "LIG") -> Optional[str]:
    """Convert RDKit mol to PDBQT string using Meeko."""
    if not MEEKO_AVAILABLE:
        print(f"[ERROR] Meeko not available for {name}")
        return None
    try:
        mol = Chem.AddHs(mol)
        if mol.GetNumConformers() == 0:
            p = AllChem.ETKDGv3()
            p.randomSeed = 42
            AllChem.EmbedMolecule(mol, p)
            AllChem.MMFFOptimizeMolecule(mol)

        prep = MoleculePreparation()
        prep.prepare(mol)
        pdbqt_str = prep.write_pdbqt_string()
        print(f"[SUCCESS] {name} converted to PDBQT ({len(pdbqt_str)} chars)")
        return pdbqt_str
    except Exception as e:
        print(f"[ERROR] Meeko conversion failed for {name}: {str(e)}")
        return None


class DockingResult:
    __slots__ = [
        "mol",
        "name",
        "vina_score",
        "gnina_affinity",
        "pose_pdb",
        "complex_pdb",
        "rmsd",
        "n_poses",
    ]

    def __init__(
        self,
        mol,
        name,
        vina_score,
        gnina_affinity,
        pose_pdb="",
        complex_pdb="",
        rmsd=0.0,
        n_poses=1,
    ):
        self.mol = mol
        self.name = name
        self.vina_score = vina_score
        self.gnina_affinity = gnina_affinity
        self.pose_pdb = pose_pdb
        self.complex_pdb = complex_pdb
        self.rmsd = rmsd
        self.n_poses = n_poses


def _vina_dock_single(
    vina_obj,
    ligand_pdbqt: str,
    exhaustiveness: int = 8,
    n_poses: int = 5,
) -> Optional[Tuple[float, str]]:
    """Dock a single ligand PDBQT and return best score + pose."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as lf:
            lf.write(ligand_pdbqt)
            lig_path = lf.name

        out_path = lig_path.replace(".pdbqt", "_out.pdbqt")
        vina_obj.set_ligand_from_file(lig_path)
        vina_obj.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        vina_obj.write_poses(out_path, n_poses=1, overwrite=True)

        energies = vina_obj.energies(n_poses=1)
        best_score = float(energies[0][0]) if energies else 0.0

        with open(out_path) as f:
            pose_str = f.read()

        os.unlink(lig_path)
        os.unlink(out_path)
        print(f"[SUCCESS] Vina docking score: {best_score:.3f}")
        return best_score, pose_str
    except Exception as e:
        print(f"[ERROR] Vina docking failed: {str(e)}")
        return None


def dock_ligands(
    mols: List[Chem.Mol],
    names: List[str],
    pdb_bytes: bytes,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    n_poses: int = 5,
    progress_bar=None,
    status_text=None,
    gnina_weights: str = "",
) -> List[DockingResult]:
    """Dock ligands to receptor using Vina + GNINA rescoring."""
    results: List[DockingResult] = []

    print(f"\n[DOCKING] ═══════════════════════════════════════════════")
    print(f"[DOCKING] Starting dock_ligands with {len(mols)} molecules")
    print(f"[DOCKING] Vina available: {VINA_AVAILABLE}")
    print(f"[DOCKING] Meeko available: {MEEKO_AVAILABLE}")
    print(f"[DOCKING] Grid center: {center}")
    print(f"[DOCKING] Box size: {box_size}")
    print(f"[DOCKING] Exhaustiveness: {exhaustiveness}, Poses: {n_poses}")
    print(f"[DOCKING] ═══════════════════════════════════════════════\n")

    if not VINA_AVAILABLE:
        print("[ERROR] AutoDock-Vina not available")
        if status_text:
            status_text.error(
                "AutoDock-Vina Python package not available. "
                "Install with: pip install vina"
            )
        return results

    with tempfile.TemporaryDirectory() as tmpdir:
        receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")

        # Write PDBQT directly (already in correct format)
        with open(receptor_pdbqt, "wb") as f:
            f.write(pdb_bytes)

        if status_text:
            status_text.info("✓ Receptor PDBQT loaded")

        cx, cy, cz = center
        sx, sy, sz = box_size

        try:
            vina_obj = Vina(sf_name="vina", verbosity=0)
            vina_obj.set_receptor(receptor_pdbqt)
            vina_obj.compute_vina_maps(center=[cx, cy, cz], box_size=[sx, sy, sz])
            print(f"[SUCCESS] Vina maps computed")
        except Exception as e:
            print(f"[ERROR] Vina setup failed: {str(e)}")
            if status_text:
                status_text.error(f"Vina error: {str(e)}")
            return results

        if status_text:
            status_text.info("✓ Vina maps computed")

        gnina_model, device = load_gnina_model(gnina_weights)

        receptor_pdb_str = pdb_bytes.decode("utf-8", errors="ignore")

        total = len(mols)
        for i, (mol, name) in enumerate(zip(mols, names)):
            print(f"\n[DOCKING] Ligand {i + 1}/{total}: {name}")
            try:
                lig_pdbqt = _mol_to_pdbqt_str(mol, name)
                if lig_pdbqt is None:
                    print(f"[SKIP] {name}: PDBQT prep failed")
                    if status_text:
                        status_text.warning(f"⚠ Skipping {name}: PDBQT prep failed")
                    if progress_bar:
                        progress_bar.progress((i + 1) / total, text=f"Skipped {name}")
                    continue

                dock_out = _vina_dock_single(
                    vina_obj,
                    lig_pdbqt,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                )
                if dock_out is None:
                    print(f"[SKIP] {name}: Vina docking failed")
                    if status_text:
                        status_text.warning(f"⚠ {name}: Vina docking failed")
                    if progress_bar:
                        progress_bar.progress((i + 1) / total, text=f"Failed {name}")
                    continue

                vina_score, pose_pdbqt = dock_out

                pdb_pose_bytes = pdbqt_to_pdb_standard(pose_pdbqt)
                if isinstance(pdb_pose_bytes, bytes):
                    pose_pdb = pdb_pose_bytes.decode("utf-8")
                else:
                    pose_pdb = pdb_pose_bytes

                complex_pdb = receptor_pdb_str.rstrip() + "\n" + pose_pdb

                gnina_aff = _gnina_rescore(gnina_model, device, mol, vina_score)

                results.append(
                    DockingResult(
                        mol=mol,
                        name=name,
                        vina_score=vina_score,
                        gnina_affinity=gnina_aff,
                        pose_pdb=pose_pdb,
                        complex_pdb=complex_pdb,
                        n_poses=n_poses,
                    )
                )
                print(f"[RESULT] {name}: ΔG={gnina_aff:.3f} kcal/mol")

            except Exception as e:
                print(f"[ERROR] {name}: {str(e)}")
                if status_text:
                    status_text.warning(f"⚠ {name}: {str(e)}")
                continue

            frac = (i + 1) / total
            if progress_bar:
                progress_bar.progress(frac, text=f"Docking {i + 1}/{total}: {name}")

    results.sort(key=lambda r: r.gnina_affinity)
    print(f"\n[DOCKING] ═══════════════════════════════════════════════")
    print(f"[DOCKING] Final results: {len(results)} ligands docked successfully")
    print(f"[DOCKING] ═══════════════════════════════════════════════\n")

    return results


def _gnina_rescore(
    model: GNINAScorer,
    device: torch.device,
    mol: Chem.Mol,
    vina_score: float,
) -> float:
    """Rescore ligand using GNINA CNN model based on coordinates."""
    try:
        positions, atom_types = _mol_to_coords_and_features(mol)
        if positions is None:
            return round(vina_score, 3)

        # Create 3D voxel grid centered on molecule centroid
        grid_size = 24
        voxel_size = 0.5  # Ångströms

        # Molecule center
        mol_center = positions.mean(axis=0)

        # Map atomic coordinates to grid indices
        grid = torch.zeros(1, 35, grid_size, grid_size, grid_size, device=device)

        for pos, atom_type in zip(positions, atom_types):
            # Relative position to center
            rel_pos = pos - mol_center

            # Convert to grid indices (center is [12, 12, 12] in 24x24x24 grid)
            grid_center = grid_size // 2
            grid_indices = (rel_pos / voxel_size + grid_center).astype(int)

            # Bounds check and channel assignment
            if all(0 <= idx < grid_size for idx in grid_indices):
                channel = min(atom_type, 34)  # 35 channels (0-34)
                x, y, z = grid_indices
                grid[0, channel, x, y, z] = 1.0

        with torch.no_grad():
            with amp.autocast() if device.type == "cuda" else torch.no_grad():
                _, affinity = model(grid)

        cnn_score = affinity.item()
        blended = 0.4 * cnn_score + 0.6 * vina_score
        return round(blended, 3)
    except Exception as e:
        print(f"[DEBUG] GNINA rescoring failed: {e}")
        return round(vina_score, 3)


def _mol_to_coords_and_features(
    mol: Chem.Mol,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract 3D coordinates and atom features from molecule."""
    try:
        conf = mol.GetConformer()
        positions = conf.GetPositions()

        # Atomic number encoding (simplified)
        atom_types = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

        return positions, atom_types
    except Exception:
        return None, None


def _fallback_score_only(
    mols: List[Chem.Mol],
    names: List[str],
    gnina_weights: str,
    progress_bar,
    status_text,
) -> List[DockingResult]:
    """Fallback: score-only mode when Vina docking unavailable."""
    if status_text:
        status_text.warning(
            "⚠ AutoDock-Vina not available — running GNINA CNN score-only mode."
        )
    model, device = load_gnina_model(gnina_weights)
    results = []
    for i, (mol, name) in enumerate(zip(mols, names)):
        try:
            score = _gnina_rescore(model, device, mol, vina_score=0.0)
            results.append(
                DockingResult(
                    mol=mol,
                    name=name,
                    vina_score=0.0,
                    gnina_affinity=score,
                )
            )
        except Exception:
            pass
        if progress_bar:
            progress_bar.progress((i + 1) / len(mols))
    results.sort(key=lambda r: r.gnina_affinity)
    return results
