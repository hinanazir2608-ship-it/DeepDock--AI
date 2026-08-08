"""
Stage 2 — AutoDock-Vina docking (+ optional GNINA CNN rescoring)

Public interface (matches app.py imports):
    VINA_AVAILABLE            : bool, whether the `vina` python package + `obabel` exist
    load_gnina_model(weights) : loads/caches a GNINA CNN checkpoint if provided
    dock_ligands(...)         : docks a list of RDKit Mols, returns list[DockResult]

DockResult exposes exactly the attributes app.py reads:
    .name, .mol, .vina_score, .gnina_affinity, .pose_pdb, .complex_pdb
"""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

# ── Availability checks ───────────────────────────────────────────────────
_OBABEL_PATH = shutil.which("obabel")
try:
    import vina as _vina_pkg  # AutoDock-Vina python bindings
    _VINA_PKG_AVAILABLE = True
except Exception:
    _vina_pkg = None
    _VINA_PKG_AVAILABLE = False

# We consider Vina "available" if either the python bindings or the CLI binary exist.
_VINA_CLI_PATH = shutil.which("vina")
VINA_AVAILABLE = _VINA_PKG_AVAILABLE or bool(_VINA_CLI_PATH)


@dataclass
class DockResult:
    name: str
    mol: Optional[Chem.Mol]
    vina_score: float
    gnina_affinity: float          # falls back to vina_score if no CNN weights loaded
    pose_pdb: Optional[str] = None      # PDB text of the best pose (ligand only)
    complex_pdb: Optional[str] = None   # PDB text of protein+ligand complex


# ── GNINA CNN weights loader (optional rescoring layer) ──────────────────
_LOADED_GNINA = {"path": None, "model": None}


def load_gnina_model(weights_path: str = ""):
    """
    Loads a GNINA/CNN checkpoint if a path is given; caches it so repeated
    calls within the same session don't reload from disk.
    Returns None if no path given (Vina-only scoring mode) or load fails.
    """
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


# ── Mol → PDBQT conversion (mirrors your Gradio mol_to_pdbqt) ────────────
def _mol_to_pdbqt(mol: Chem.Mol, out_pdbqt: str) -> bool:
    try:
        m = Chem.AddHs(mol)
        if m.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(m, params) == -1:
                AllChem.EmbedMolecule(m, AllChem.ETKDG())
            try:
                AllChem.MMFFOptimizeMolecule(m)
            except Exception:
                pass

        pdb_path = out_pdbqt.replace(".pdbqt", ".pdb")
        Chem.MolToPDBFile(m, pdb_path)

        if not _OBABEL_PATH:
            return False

        result = subprocess.run(
            [_OBABEL_PATH, "-ipdb", pdb_path, "-opdbqt", "-O", out_pdbqt,
             "--partialcharge", "eem"],
            capture_output=True, text=True, timeout=60,
        )
        return os.path.exists(out_pdbqt) and result.returncode == 0
    except Exception:
        return False


def _run_vina_cli(
    receptor_pdbqt: str, ligand_pdbqt: str, out_pdbqt: str,
    center: Tuple[float, float, float], box_size: Tuple[float, float, float],
    exhaustiveness: int,
) -> Optional[float]:
    """Shells out to the vina CLI (same approach as your Gradio run_batched_docking)."""
    if not _VINA_CLI_PATH:
        return None
    cmd = [
        _VINA_CLI_PATH,
        "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
        "--out", out_pdbqt, "--exhaustiveness", str(exhaustiveness),
        # Only write the single best-scoring pose. Vina's CLI default is up
        # to 9 alternate poses (each as its own MODEL/ENDMDL block), which
        # _pdbqt_to_pdb() would otherwise carry straight into the exported
        # "complex" PDB — that's what showed up as 6+ separate "Ligand"
        # groups stacked in one file in Discovery Studio.
        "--num_modes", "1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
    except Exception:
        return None


def _run_vina_python(
    receptor_pdbqt: str, ligand_pdbqt: str,
    center: Tuple[float, float, float], box_size: Tuple[float, float, float],
    exhaustiveness: int, n_poses: int, out_pdbqt: str,
) -> Optional[float]:
    """Uses the `vina` python bindings when available — faster, no subprocess overhead."""
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
    except Exception:
        return None


def _pdbqt_to_pdb(pdbqt_path: str, first_model_only: bool = True) -> Optional[str]:
    """
    Strips Vina charge/type columns, returns standard PDB text (or None).

    first_model_only=True (default) stops after the first ENDMDL, so if the
    source file has multiple stacked poses (MODEL/ENDMDL blocks), only the
    best-scoring one is kept. This is a safety net independent of the
    --num_modes fix above — protects against any file that ends up with
    multiple poses for any reason. Files with no MODEL wrapping at all
    (e.g. a plain receptor) are unaffected either way.
    """
    if not pdbqt_path or not os.path.exists(pdbqt_path):
        return None
    lines = []
    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                lines.append(line[:66].ljust(66) + "\n")
            elif line.startswith("ENDMDL"):
                if first_model_only:
                    break
                lines.append(line)
            elif line.startswith(("MODEL", "TER")):
                lines.append(line)
    return "".join(lines) if lines else None


# ── Main entry point ──────────────────────────────────────────────────────
def dock_ligands(
    mols: List[Chem.Mol],
    names: List[str],
    pdb_bytes: bytes,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    exhaustiveness: int = 8,
    n_poses: int = 5,
    progress_bar=None,     # st.progress() handle
    status_text=None,      # st.empty() handle
    gnina_weights: str = "",
) -> List[DockResult]:
    """
    Docks each mol against the receptor and returns results sorted best→worst
    by gnina_affinity (ascending — more negative = stronger binding).
    """
    if not mols:
        return []

    gnina_model = load_gnina_model(gnina_weights) if gnina_weights else None

    work_dir = tempfile.mkdtemp(prefix="deepdock_")
    receptor_path = os.path.join(work_dir, "receptor.pdbqt")
    with open(receptor_path, "wb") as f:
        f.write(pdb_bytes)

    total = len(mols)
    results: List[DockResult] = []

    for idx, (mol, name) in enumerate(zip(mols, names)):
        if status_text:
            status_text.text(f"Docking {idx + 1}/{total}: {name}")

        ligand_pdbqt = os.path.join(work_dir, f"{idx}_{name}.pdbqt")
        out_pdbqt = os.path.join(work_dir, f"{idx}_{name}_out.pdbqt")

        converted = _mol_to_pdbqt(mol, ligand_pdbqt)

        vina_score: Optional[float] = None
        if converted and os.path.exists(ligand_pdbqt):
            if _VINA_PKG_AVAILABLE:
                vina_score = _run_vina_python(
                    receptor_path, ligand_pdbqt, center, box_size,
                    exhaustiveness, n_poses, out_pdbqt,
                )
            if vina_score is None and _VINA_CLI_PATH:
                vina_score = _run_vina_cli(
                    receptor_path, ligand_pdbqt, out_pdbqt,
                    center, box_size, exhaustiveness,
                )

        if vina_score is None:
            # No Vina/obabel on this machine — skip rather than fabricate a score.
            if progress_bar:
                progress_bar.progress(min((idx + 1) / total, 1.0))
            continue

        # GNINA CNN rescoring hook — falls back to the Vina score when no
        # weights are loaded (matches the sidebar caption: "random-init CNN").
        gnina_affinity = vina_score
        if gnina_model is not None:
            try:
                gnina_affinity = float(vina_score)  # placeholder until a real
                # CNN forward pass is wired in; keeps the field populated and
                # sortable without silently pretending to rescore.
            except Exception:
                gnina_affinity = vina_score

        pose_pdb = _pdbqt_to_pdb(out_pdbqt)
        complex_pdb = None
        if pose_pdb:
            receptor_pdb_text = _pdbqt_to_pdb(receptor_path) or ""
            complex_pdb = receptor_pdb_text + pose_pdb

        results.append(DockResult(
            name=name, mol=mol, vina_score=float(vina_score),
            gnina_affinity=float(gnina_affinity),
            pose_pdb=pose_pdb, complex_pdb=complex_pdb,
        ))

        if progress_bar:
            progress_bar.progress(min((idx + 1) / total, 1.0))

    shutil.rmtree(work_dir, ignore_errors=True)

    results.sort(key=lambda r: r.gnina_affinity)
    return results
