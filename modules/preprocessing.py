import os
import sys

# Ensure correct working directory
target_dir = "/kaggle/working/DeepDock--AI"
if os.path.exists(target_dir):
    os.chdir(target_dir)

preprocessing_code = '''"""
Preprocessing Module with Robust OpenBabel & Fail-Safe PDBQT Conversion
"""
from __future__ import annotations
import os
import io
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def clean_pdbqt_string(pdbqt_str: str) -> str:
    """Strips unwanted headers to ensure AutoDock Vina compatibility."""
    allowed = ("ATOM", "HETATM", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "MODEL", "ENDMDL")
    lines = pdbqt_str.splitlines()
    clean = [l for l in lines if l.strip().startswith(allowed)]
    return "\\n".join(clean) + "\\n"


def mol_to_pdbqt(mol: Chem.Mol, out_path: str) -> bool:
    """
    Converts RDKit Mol object to PDBQT.
    Tries Open Babel first; if it fails, falls back to sanitized structural dump.
    """
    try:
        # Ensure 3D coords and Hydrogens exist
        mol_h = Chem.AddHs(mol, addCoords=True)
        if mol_h.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == -1:
                AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
            try:
                AllChem.MMFFOptimizeMolecule(mol_h)
            except Exception:
                pass

        # Write temporary SDF
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w") as tmp:
            writer = Chem.SDWriter(tmp.name)
            writer.write(mol_h)
            writer.close()
            tmp_sdf = tmp.name

        # --- Method 1: OpenBabel Execution ---
        cmd = ["obabel", tmp_sdf, "-O", out_path, "--gen3d", "-p", "7.4", "-partialcharge", "gasteiger"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Cleanup temp SDF
        if Path(tmp_sdf).exists():
            Path(tmp_sdf).unlink()

        # Check if output generated and non-empty
        if res.returncode == 0 and Path(out_path).exists() and os.path.getsize(out_path) > 0:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            sanitized = clean_pdbqt_string(content)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(sanitized)
            return True

        # --- Method 2: OpenBabel Fallback (Simplified Flag Command) ---
        cmd_alt = ["obabel", "-isdf", tmp_sdf if Path(tmp_sdf).exists() else "", "-opdbqt", "-O", out_path]
        res_alt = subprocess.run(cmd_alt, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if Path(out_path).exists() and os.path.getsize(out_path) > 0:
            return True

        logger.error(f"OpenBabel conversion failed: {res.stderr}")
        return False

    except Exception as e:
        logger.error(f"Error in mol_to_pdbqt: {e}")
        return False


def load_ligands_from_bytes(sdf_bytes: bytes) -> Tuple[List[Chem.Mol], List[str]]:
    """Loads molecules and extracts/generates CID names from SDF bytes."""
    stream = io.BytesIO(sdf_bytes)
    suppl = Chem.ForwardSDMolSupplier(stream, removeHs=False, sanitize=True)
    mols, names = [], []
    idx = 0
    for mol in suppl:
        idx += 1
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        name = str(
            props.get("CID") or 
            props.get("PUBCHEM_COMPOUND_CID") or 
            props.get("CHEMBL_ID") or 
            props.get("_Name") or 
            f"ligand_{idx}"
        ).strip()
        if not name or name == "None":
            name = f"ligand_{idx}"

        mol.SetProp("_Name", name)
        mols.append(mol)
        names.append(name)
    return mols, names


def ensure_3d_coords(mols: List[Chem.Mol]) -> List[Chem.Mol]:
    """Generates 3D conformations if missing."""
    out = []
    for mol in mols:
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) == -1:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            mol = Chem.RemoveHs(mol)
        out.append(mol)
    return out
'''

os.makedirs("modules", exist_ok=True)
with open("modules/preprocessing.py", "w") as f:
    f.write(preprocessing_code)

print("✅ Patched modules/preprocessing.py with robust OpenBabel conversion!")
