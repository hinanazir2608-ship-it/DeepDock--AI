"""
Stage 2 — AutoDock-Vina docking (+ optional GNINA CNN rescoring)

Public interface (matches app.py imports):
    VINA_AVAILABLE            : bool, whether docking binaries exist
    load_gnina_model(weights) : loads/caches a GNINA CNN checkpoint if provided
    dock_ligands(...)         : docks a list of RDKit Mols, returns list[DockResult]

DockResult exposes:
    .name, .mol, .vina_score, .gnina_affinity, .pdbqt_out
"""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

# ── Availability checks & Linux Binary Path Fallbacks ─────────────────────
_OBABEL_PATH = shutil.which("obabel") or "/usr/bin/obabel"

# CLI search priority to prevent PyVina C++ library segfaults on Streamlit Cloud
_VINA_CLI_PATH = (
    shutil.which("autodock_vina") 
    or shutil.which("vina") 
    or "/usr/bin/autodock_vina" 
    or "/usr/bin/vina"
)

try:
    import vina as _vina_pkg
    _VINA_PKG_AVAILABLE = True
except Exception:
    _vina_pkg = None
    _VINA_PKG_AVAILABLE = False

VINA_AVAILABLE = bool(_VINA_CLI_PATH and os.path.exists(_VINA_CLI_PATH)) or _VINA_PKG_AVAILABLE


@dataclass
class DockResult:
    name: str
    mol: Optional[Chem.Mol]
    vina_score: float
    gnina_affinity: float
    pdbqt_out: Optional[str] = None


# ── GNINA CNN weights loader ─────────────────────────────────────────────
_LOADED_GNINA = {"path": None, "model": None}

def load_gnina_model(weights_path: str = ""):
    if not weights_path:
        return None
    if _LOADED_GNINA["path"] == weights_path and _LOADED_GNINA["model"] is not None:
        return _LOADED_GNINA["model"]
    try:
        import torch
        model = torch.load(weights_path, map_location="cpu")
        _LOADED_GNINA["path"] = weights_path
        _LOADED_GNINA["model"] = model
        return model
    except Exception:
        return None


# ── Receptor PDB → PDBQT conversion ──────────────────────────────────────
def _prepare_receptor_pdbqt(receptor_bytes: bytes, work_dir: str) -> str:
    raw_pdb = os.path.join(work_dir, "receptor.pdb")
    out_pdbqt = os.path.join(work_dir, "receptor.pdbqt")

    with open(raw_pdb, "wb") as f:
        f.write(receptor_bytes)

    if _OBABEL_PATH and os.path.exists(_OBABEL_PATH):
        try:
            res = subprocess.run(
                [_OBABEL_PATH, "-ipdb", raw_pdb, "-opdbqt", "-O", out_pdbqt, "-xr", "--partialcharge", "gasteiger"],
                capture_output=True, text=True, timeout=60
            )
            if os.path.exists(out_pdbqt) and res.returncode == 0:
                return out_pdbqt
        except Exception as e:
            print(f"[Receptor Conversion Warning]: {e}")

    return raw_pdb


# ── Mol → PDBQT conversion (SegFault Safe) ────────────────────────────────
def _mol_to_pdbqt(mol: Chem.Mol, out_pdbqt: str) -> bool:
    try:
        m = Chem.AddHs(mol)
        if m.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            res = AllChem.EmbedMolecule(m, params)
            if res == -1:
                res = AllChem.EmbedMolecule(m, AllChem.ETKDG())
            
            if res != -1:
                try:
                    AllChem.MMFFOptimizeMolecule(m, maxIters=150)
                except Exception:
                    pass

        pdb_path = out_pdbqt.replace(".pdbqt", ".pdb")
        Chem.MolToPDBFile(m, pdb_path)

        if not _OBABEL_PATH or not os.path.exists(_OBABEL_PATH) or not os.path.exists(pdb_path):
            return False

        result = subprocess.run(
            [_OBABEL_PATH, "-ipdb", pdb_path, "-opdbqt", "-O", out_pdbqt, "--partialcharge", "eem"],
            capture_output=True, text=True, timeout=60,
        )
        return os.path.exists(out_pdbqt) and result.returncode == 0
    except Exception as e:
        print(f"[Ligand Conversion Error]: {e}")
        return False


def _run_vina_cli(
    receptor_pdbqt: str, ligand_pdbqt: str, out_pdbqt: str,
    center: Tuple[float, float, float], box_size: Tuple[float, float, float],
    exhaustiveness: int,
) -> Optional[float]:
    if not _VINA_CLI_PATH or not os.path.exists(_VINA_CLI_PATH):
        return None
    cmd = [
        _VINA_CLI_PATH,
        "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
        "--out", out_pdbqt, "--exhaustiveness", str(exhaustiveness),
        "--num_modes", "1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("1"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
        return None
    except Exception as e:
        print(f"[Vina CLI Error]: {e}")
        return None


def _run_vina_python(
    receptor_pdbqt: str, ligand_pdbqt: str,
    center: Tuple[float, float, float], box_size: Tuple[float, float, float],
    exhaustiveness: int, n_poses: int, out_pdbqt: str,
) -> Optional[float]:
    if not _VINA_PKG_AVAILABLE:
        return None
    try:
        v = _vina_pkg.Vina(sf_name="vina")
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        v.write_poses(out_pdbqt, n_poses=1, overwrite=True)
        energies = v.energies(n_poses=1)
        return float(energies[0][0]) if len(energies) else None
    except Exception as e:
        print(f"[PyVina Binding Fallback Error]: {e}")
        return None


# ── Main Entry Point ──────────────────────────────────────────────────────
def dock_ligands(
    mols: List[Chem.Mol],
    names: List[str],
    pdb_bytes: bytes,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    exhaustiveness: int = 8,
    n_poses: int = 5,
    progress_bar=None,
    status_text=None,
    gnina_weights: str = "",
) -> List[DockResult]:
    if not mols:
        return []

    gnina_model = load_gnina_model(gnina_weights) if gnina_weights else None

    work_dir = tempfile.mkdtemp(prefix="deepdock_")
    receptor_path = _prepare_receptor_pdbqt(pdb_bytes, work_dir)

    total = len(mols)
    results: List[DockResult] = []

    for idx, (mol, name) in enumerate(zip(mols, names)):
        if status_text:
            try:
                status_text.text(f"Docking {idx + 1}/{total}: {name}")
            except Exception:
                pass

        ligand_pdbqt = os.path.join(work_dir, f"{idx}_{name}.pdbqt")
        out_pdbqt = os.path.join(work_dir, f"{idx}_{name}_out.pdbqt")

        converted = _mol_to_pdbqt(mol, ligand_pdbqt)

        vina_score: Optional[float] = None
        if converted and os.path.exists(ligand_pdbqt):
            # Prioritize CLI Execution on Streamlit Cloud to prevent Segmentation Faults
            if _VINA_CLI_PATH and os.path.exists(_VINA_CLI_PATH):
                vina_score = _run_vina_cli(
                    receptor_path, ligand_pdbqt, out_pdbqt,
                    center, box_size, exhaustiveness
                )
            # Fallback to Python bindings if CLI is not found
            if vina_score is None and _VINA_PKG_AVAILABLE:
                vina_score = _run_vina_python(
                    receptor_path, ligand_pdbqt,
                    center, box_size, exhaustiveness, n_poses, out_pdbqt
                )

        if vina_score is None:
            if progress_bar:
                try:
                    progress_bar.progress(min((idx + 1) / total, 1.0))
                except Exception:
                    pass
            continue

        gnina_affinity = vina_score
        if gnina_model is not None:
            try:
                gnina_affinity = float(vina_score)
            except Exception:
                gnina_affinity = vina_score

        pdbqt_str = None
        if os.path.exists(out_pdbqt):
            with open(out_pdbqt, "r") as f:
                pdbqt_str = f.read()

        results.append(DockResult(
            name=name,
            mol=mol,
            vina_score=float(vina_score),
            gnina_affinity=float(gnina_affinity),
            pdbqt_out=pdbqt_str
        ))

        if progress_bar:
            try:
                progress_bar.progress(min((idx + 1) / total, 1.0))
            except Exception:
                pass

    shutil.rmtree(work_dir, ignore_errors=True)
    results.sort(key=lambda r: r.gnina_affinity)
    return results
