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

from modules.export import pdbqt_to_pdb_standard

try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

try:
    import meeko
    from meeko import MoleculePreparation, PDBQTMolecule
    MEEKO_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG IMPORT ERROR] Meeko import failed: {e}")
    MEEKO_AVAILABLE = False


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
    IN_CHANNELS  = 35
    GRID_SIZE    = 24

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            GNINAConvBlock(self.IN_CHANNELS, 32, pool=False),
            GNINAConvBlock(32, 64,  pool=True),
            GNINAConvBlock(64, 128, pool=True),
            GNINAConvBlock(128, 256, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.trunk = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.pose_head     = nn.Linear(256, 1)
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
        pose  = torch.sigmoid(self.pose_head(x)).squeeze(-1)
        affin = (self.affinity_head(x).squeeze(-1) * -10.0) - 5.0
        return pose, affin


@st.cache_resource(show_spinner="Loading GNINA scoring model…")
def load_gnina_model(weights_path: str = "") -> Tuple[GNINAScorer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GNINAScorer().to(device).eval()
    if weights_path and Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    return model, device


# modules/docking.py me _mol_to_pdbqt_str method check kar lein:

def _mol_to_pdbqt_str(mol: Chem.Mol, name: str = "LIG") -> Optional[str]:
    if not MEEKO_AVAILABLE:
        print("[DEBUG] Meeko is not imported properly!")
        return None
    try:
        mol = Chem.Mol(mol)  # Copy mol
        mol.SetProp("_Name", str(name)) # 🔹 Preserve CID name in copy
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)

        # Ensure 3D Conformer exists
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.useRandomCoordinates = True
            params.maxAttempts = 500
            res = AllChem.EmbedMolecule(mol, params)
            if res != 0:
                print(f"[DEBUG] 3D embedding failed for {name}")
                return None

        # Force Energy Minimization (MMFF or UFF fallback)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass  # Proceed even if force field optimization fails

        # Modern Meeko v0.5+ method (No deprecation warnings)
        try:
            pdbqt_mol = PDBQTMolecule.from_rdkit(mol)
            return pdbqt_mol.write_pdbqt_string()
        except (ImportError, AttributeError):
            # Legacy Fallback
            prep = MoleculePreparation()
            prep.prepare(mol)
            return prep.write_pdbqt_string()

    except Exception as e:
        print(f"[DEBUG] Meeko Conversion Exception for {name}: {e}")
        return None


class DockingResult:
    __slots__ = ["mol", "name", "vina_score", "gnina_affinity",
                 "pose_pdb", "complex_pdb", "rmsd", "n_poses"]

    def __init__(self, mol, name, vina_score, gnina_affinity,
                 pose_pdb="", complex_pdb="", rmsd=0.0, n_poses=1):
        self.mol            = mol
        self.name           = name
        self.vina_score     = vina_score
        self.gnina_affinity = gnina_affinity
        self.pose_pdb        = pose_pdb
        self.complex_pdb    = complex_pdb
        self.rmsd            = rmsd
        self.n_poses        = n_poses


def _vina_dock_single(
    vina_obj,
    ligand_pdbqt: str,
    exhaustiveness: int = 8,
    n_poses: int = 5,
) -> Optional[Tuple[float, str]]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as lf:
            lf.write(ligand_pdbqt)
            lig_path = lf.name

        out_path = lig_path.replace(".pdbqt", "_out.pdbqt")
        vina_obj.set_ligand_from_file(lig_path)
        vina_obj.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        vina_obj.write_poses(out_path, n_poses=1, overwrite=True)

        # Fix: Safe array evaluation for numpy arrays
        energies = vina_obj.energies(n_poses=1)
        best_score = float(energies[0][0]) if len(energies) > 0 else 0.0

        with open(out_path) as f:
            pose_str = f.read()

        os.unlink(lig_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
            
        return best_score, pose_str
    except Exception as e:
        print(f"[DEBUG] Vina dock single error: {e}")
        return None


def _gnina_rescore(
    model:      GNINAScorer,
    device:     torch.device,
    mol:        Chem.Mol,
    vina_score: float,
) -> float:
    try:
        from rdkit.Chem import rdMolDescriptors
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.array(fp)
        grid = torch.zeros(1, 35, 24, 24, 24, device=device)
        for bit_idx in np.nonzero(arr)[0][:35 * 24]:
            ch  = bit_idx % 35
            pos = bit_idx % 24
            grid[0, ch, pos, pos, pos] = 1.0

        with torch.no_grad():
            with amp.autocast() if device.type == "cuda" else torch.no_grad():
                _, affinity = model(grid)
        cnn_score = affinity.item()
        blended   = 0.4 * cnn_score + 0.6 * vina_score
        return round(blended, 3)
    except Exception:
        return round(vina_score, 3)


def _fallback_score_only(
    mols:        List[Chem.Mol],
    names:       List[str],
    gnina_weights: str,
    progress_bar,
    status_text,
) -> List[DockingResult]:
    if status_text:
        status_text.warning(
            "AutoDock-Vina not available — running GNINA CNN score-only mode."
        )
    model, device = load_gnina_model(gnina_weights)
    results = []
    for i, (mol, name) in enumerate(zip(mols, names)):
        try:
            score = _gnina_rescore(model, device, mol, vina_score=0.0)
            results.append(DockingResult(
                mol=mol, name=name,
                vina_score=0.0, gnina_affinity=score,
            ))
        except Exception:
            pass
        if progress_bar:
            progress_bar.progress((i + 1) / len(mols))
    results.sort(key=lambda r: r.gnina_affinity)
    return results


def dock_ligands(
    mols:           List[Chem.Mol],
    names:          List[str],
    pdb_bytes:      Union[bytes, str],
    center:         Tuple[float, float, float],
    box_size:       Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    n_poses:         int = 5,
    progress_bar    = None,
    status_text     = None,
    gnina_weights:  str = "",
) -> List[DockingResult]:
    results: List[DockingResult] = []

    if not VINA_AVAILABLE:
        if status_text:
            status_text.error(
                "AutoDock-Vina Python package not available. "
                "Install with: pip install vina"
            )
        return _fallback_score_only(mols, names, gnina_weights, progress_bar, status_text)

    with tempfile.TemporaryDirectory() as tmpdir:
        receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")

        # Safely convert pdb_bytes/str to standard string
        if isinstance(pdb_bytes, bytes):
            receptor_pdb_str = pdb_bytes.decode("utf-8", errors="ignore")
        else:
            receptor_pdb_str = str(pdb_bytes)

        with open(receptor_pdbqt, "w", encoding="utf-8") as f:
            f.write(receptor_pdb_str)

        if status_text:
            status_text.info("✓ Receptor PDBQT ready (User provided)") 

        center_list = [float(c) for c in center]
        box_size_list = [float(s) for s in box_size]

        vina_obj = Vina(sf_name="vina", verbosity=0)
        vina_obj.set_receptor(receptor_pdbqt)
        vina_obj.compute_vina_maps(center=center_list, box_size=box_size_list)

        gnina_model, device = load_gnina_model(gnina_weights)

        total = len(mols)
        for i, (mol, name) in enumerate(zip(mols, names)):
            try:
                lig_pdbqt = _mol_to_pdbqt_str(mol, name)
                if lig_pdbqt is None:
                    print(f"[DEBUG] Meeko failed to convert ligand: {name}")
                    if status_text:
                        status_text.warning(f"Skipping {name}: PDBQT prep failed")
                    continue

                dock_out = _vina_dock_single(
                    vina_obj, lig_pdbqt,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                )
                if dock_out is None:
                    print(f"[DEBUG] Vina execution failed for ligand: {name}")
                    continue

                vina_score, pose_pdbqt = dock_out
                pdb_pose_out = pdbqt_to_pdb_standard(pose_pdbqt)

                # Safely handle string vs bytes for pose
                if isinstance(pdb_pose_out, bytes):
                    pose_pdb = pdb_pose_out.decode("utf-8", errors="ignore")
                else:
                    pose_pdb = str(pdb_pose_out)

                complex_pdb = receptor_pdb_str.rstrip() + "\n" + pose_pdb

                gnina_aff = _gnina_rescore(gnina_model, device, mol, vina_score)

                results.append(DockingResult(
                    mol           = mol,
                    name          = name,
                    vina_score    = vina_score,
                    gnina_affinity= gnina_aff,
                    pose_pdb      = pose_pdb,
                    complex_pdb   = complex_pdb,
                    n_poses       = n_poses,
                ))

            except Exception as e:
                print(f"[DEBUG] Exception for {name}: {e}")
                if status_text:
                    status_text.warning(f"{name}: {e}")
                continue

            frac = (i + 1) / total
            if progress_bar:
                progress_bar.progress(frac, text=f"Docking {i+1}/{total}: {name}")

    results.sort(key=lambda r: r.gnina_affinity)
    return results
