"""
Preprocessing: SDF ligand loading + PDB active-site parsing + conf.txt grid parsing + PDBQT Conversion.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import io
import subprocess
import tempfile
import logging

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)


# ── PDBQT Conversion & Sanitization Helper ────────────────────────────
def clean_pdbqt_string(pdbqt_str: str) -> str:
    """
    Sanitize PDBQT content by stripping illegal tags (COMPND, HEADER, etc.)
    that crash AutoDock Vina's C++ parser.
    """
    allowed_keywords = ("ATOM", "HETATM", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
    lines = pdbqt_str.splitlines()
    clean_lines = [line for line in lines if line.strip().startswith(allowed_keywords)]
    return "\n".join(clean_lines) + "\n"


def mol_to_pdbqt(mol: Chem.Mol, out_path: str) -> bool:
    """
    Converts an RDKit Mol object into a valid AutoDock Vina PDBQT file using OpenBabel.
    Ensures sanitization of COMPND / invalid header tags.
    """
    try:
        # Prepare mol with hydrogens
        mol_h = Chem.AddHs(mol, addCoords=True)
        
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w") as tmp_sdf:
            writer = Chem.SDWriter(tmp_sdf.name)
            writer.write(mol_h)
            writer.close()
            tmp_sdf_path = tmp_sdf.name

        # Convert SDF to PDBQT using obabel CLI
        cmd = ["obabel", tmp_sdf_path, "-O", out_path, "-vh"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Cleanup temp SDF file
        if Path(tmp_sdf_path).exists():
            Path(tmp_sdf_path).unlink()

        if res.returncode == 0 and Path(out_path).exists():
            # Sanitize generated PDBQT
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            sanitized_content = clean_pdbqt_string(content)
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(sanitized_content)
            
            return True
        else:
            logger.error(f"Obabel conversion failed: {res.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error converting molecule to PDBQT: {e}")
        return False


# ── Ligand SDF loading ──────────────────────────────────────────────────

def load_ligands_from_bytes(sdf_bytes: bytes) -> Tuple[List[Chem.Mol], List[str]]:
    stream = io.BytesIO(sdf_bytes)
    suppl = Chem.ForwardSDMolSupplier(stream, removeHs=False, sanitize=True)

    mols: List[Chem.Mol] = []
    names: List[str] = []
    idx = 0

    for mol in suppl:
        idx += 1
        if mol is None:
            continue

        # Extract CID or fallback to _Name
        props = mol.GetPropsAsDict()
        name = (
            props.get("CID") or
            props.get("PUBCHEM_COMPOUND_CID") or
            props.get("CHEMBL_ID") or
            props.get("_Name") or
            f"ligand_{idx}"
        )
        name = str(name).strip()
        if not name or name == "None":
            name = f"ligand_{idx}"

        # Molecule object ke ander bhi '_Name' update rakhein
        mol.SetProp("_Name", name)

        mols.append(mol)
        names.append(name)

    return mols, names


def ensure_3d_coords(mols: List[Chem.Mol]) -> List[Chem.Mol]:
    """
    Ensure each mol has 3D coordinates; generate via ETKDG if missing.
    """
    out = []
    for mol in mols:
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            mol = Chem.RemoveHs(mol)
        out.append(mol)
    return out


# ── PDB active-site parsing ─────────────────────────────────────────────
def parse_pdb_binding_site(pdb_bytes: bytes) -> Dict:
    """
    Extract the centroid of the binding site from a PDB file.
    Heuristic: if HETATM records (non-water small molecules) are present,
    use their geometric centroid; otherwise fall back to all Cα atoms.

    Returns a dict with keys: centroid (x,y,z), residue_range, n_atoms, raw_text
    """
    text = pdb_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    hetatm_coords: List[Tuple[float, float, float]] = []
    ca_coords: List[Tuple[float, float, float]] = []

    for line in lines:
        record = line[:6].strip()
        if record == "HETATM":
            res_name = line[17:20].strip()
            if res_name in ("HOH", "WAT", "DOD", "H2O"):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                hetatm_coords.append((x, y, z))
            except (ValueError, IndexError):
                pass
        elif record == "ATOM":
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ca_coords.append((x, y, z))
                except (ValueError, IndexError):
                    pass

    coords = hetatm_coords if hetatm_coords else ca_coords
    if not coords:
        return {"centroid": (0.0, 0.0, 0.0), "n_atoms": 0, "source": "none", "raw_text": text[:500]}

    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)

    return {
        "centroid": (round(cx, 3), round(cy, 3), round(cz, 3)),
        "n_atoms": len(coords),
        "source": "HETATM" if hetatm_coords else "Cα centroid",
        "raw_text": text[:500],
    }


# ── conf.txt grid parsing ───────────────────────────────────────────────
def parse_conf_txt(conf_bytes: bytes) -> Dict:
    """
    Parse an AutoDock-Vina style conf.txt for grid center and size.
    """
    text = conf_bytes.decode("utf-8", errors="ignore")
    kv: Dict[str, float] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip().lower().replace("-", "_")
            try:
                kv[key] = float(val.strip())
            except ValueError:
                pass

    center = (
        kv.get("center_x", 0.0),
        kv.get("center_y", 0.0),
        kv.get("center_z", 0.0),
    )
    size = (
        kv.get("size_x", 20.0),
        kv.get("size_y", 20.0),
        kv.get("size_z", 20.0),
    )
    extra = {
        k: v for k, v in kv.items() 
        if k not in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    }

    return {"center": center, "size": size, "extra": extra}
