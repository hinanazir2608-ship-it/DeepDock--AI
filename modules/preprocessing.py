# modules/preprocessing.py
# FIXED VERSION: Ensures load_ligands_from_bytes returns (mols, names)
# to resolve "too many values to unpack (expected 2)" in app.py

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import io

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ── CID extraction ──────────────────────────────────────────────────────
# Common SDF property tags that carry a compound identifier (PubChem CID,
# CMNPD marine-metabolite ID, etc). Checked in order; first match wins.
CID_PROPERTY_KEYS = (
    "PUBCHEM_COMPOUND_CID",
    "PUBCHEM_CID",
    "CID",
    "CMNPD_ID",
    "CMNPD_CID",
    "COMPOUND_ID",
    "Compound_ID",
    "compound_id",
    "ID",
)


def _extract_cid(mol: Chem.Mol, idx: int, name: str) -> str:
    """
    Look for a dedicated compound-ID property on the mol (PubChem CID,
    CMNPD ID, etc). Falls back to the ligand name (many CMNPD/PubChem SDF
    exports use the ID as the title itself), then to a generated placeholder.
    """
    props = mol.GetPropsAsDict()
    for key in CID_PROPERTY_KEYS:
        val = props.get(key)
        if val not in (None, ""):
            return str(val).strip()
    # Logic for when generic name is not sufficient seed
    if name and not name.startswith(
            "ligand_") and not name.startswith("Compound_"):
        return name
    return f"CID_{idx}"

def clean_and_prepare_receptor(target_file_path, output_pdb_path, output_pdbqt_path):
    """
    Cleans target PDB file, removes water and heteroatoms, 
    and outputs strictly formatted PDBQT compatible with GNINA/AutoDock.
    """
    try:
        with open(target_file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        cleaned_pdb_lines = []
        pdbqt_lines = []
        current_chain = None

        for line in lines:
            # Skip water molecules
            if any(wat in line for wat in ["HOH", "WAT"]):
                continue
            
            # Skip heteroatoms if you want pure protein residues
            if line.startswith("HETATM"):
                continue

            if line.startswith("ATOM"):
                chain_id = line[21] if len(line) > 21 else 'A'
                if current_chain is not None and chain_id != current_chain:
                    cleaned_pdb_lines.append("TER\n")
                    pdbqt_lines.append("TER\n")
                current_chain = chain_id

                # 1. Standard PDB line for visualization
                std_line = line[:66].ljust(66) + "\n"
                cleaned_pdb_lines.append(std_line)

                # 2. Extract element symbol to map to valid AutoDock atom types
                element = ""
                if len(line) >= 78:
                    element = line[76:78].strip().upper()
                
                if not element or not element.isalpha():
                    atom_name = line[12:16].strip()
                    element = ''.join([c for c in atom_name if c.isalpha()]).upper()[:1]

                # Map to standard AutoDock atom types
                if element == 'C':
                    ad_type = 'A'
                elif element == 'N':
                    ad_type = 'N'
                elif element == 'O':
                    ad_type = 'OA'
                elif element == 'S':
                    ad_type = 'SA'
                elif element == 'P':
                    ad_type = 'P'
                elif element == 'H':
                    atom_name = line[12:16].strip()
                    ad_type = 'HD' if atom_name.startswith('H') else 'H'
                else:
                    ad_type = element if len(element) == 1 else 'A'

                # 3. Strictly formatted PDBQT line (Columns 0-54: standard coords/prefix, 
                # followed by exact AutoDock spacing for charge and atom type)
                prefix = line[:54]  # ATOM serial, name, resName, chain, resSeq, X, Y, Z
                
                # Standard AutoDock PDBQT format: 
                # [Prefix up to coords] + [Charge (6 spaces, e.g., "  0.00")] + [Spacing] + [Atom Type]
                pdbqt_line = f"{prefix}  0.0000 {ad_type:>2}\n"
                pdbqt_lines.append(pdbqt_line)

            elif line.startswith(("TER", "HEADER", "REMARK", "MODEL", "ENDMDL")):
                cleaned_pdb_lines.append(line if line.endswith("\n") else line + "\n")
                pdbqt_lines.append(line if line.endswith("\n") else line + "\n")

        with open(output_pdb_path, "w", encoding="utf-8") as f:
            f.writelines(cleaned_pdb_lines)

        with open(output_pdbqt_path, "w", encoding="utf-8") as f:
            f.writelines(pdbqt_lines)

        return True
    except Exception as e:
        print(f"[ERROR] Receptor cleaning failed: {e}")
        return False
        
# ── Ligand SDF loading (FIXED Return Type) ───────────────────────────────────
def load_ligands_from_bytes(
    sdf_bytes: bytes,
) -> Tuple[List[Chem.Mol], List[str]]:  # CRITICAL FIX: Returns only 2 values
    """
    Parse an SDF file from raw bytes using SDMolSupplier.
    Returns: Tuple of (List of RDKit Molecules, List of Names/CIDs)

    This function explicitly extracts names during parsing to ensure the
    expected tuple format is returned, fixing the ValueError crash in app.py.
    """
    # SDMolSupplier needs a file path or stream; use ForwardSDMolSupplier for
    # bytes
    stream = io.BytesIO(sdf_bytes)
    # Using sanitize=True ensures molecules are valid
    suppl = Chem.ForwardSDMolSupplier(stream, removeHs=False, sanitize=True)

    mols: List[Chem.Mol] = []
    names: List[str] = []
    # cids:  List[str]      = [] # Not required in main app return

    idx = 0

    try:
        # Suppress RDKit logs for cleaner output
        Chem.WrapLogs()

        for mol in suppl:
            idx += 1
            if mol is None:
                continue

            # Name extraction logic (matches app.py fallback and ensures
            # uniqueness)
            name = mol.GetPropsAsDict().get("_Name", "").strip()
            if not name:
                name = f"Compound_{idx}"

            # The app logic also extracts CID information and preserves it
            # within the molecule object.
            cid = _extract_cid(mol, idx, name)
            # CRITICAL: Preserve CID/NamesSeed internally for filtering stages
            mol.SetProp("Compound CID / Name", cid)
            # app.py requires consistent standard property _Name as well
            mol.SetProp("_Name", cid)

            mols.append(mol)
            names.append(cid)  # Using CID as standard unique name reference

    except Exception as e:
        print(f"[ERROR PREPROCESSING] Loading ligands failed: {e}")
        # Return what we managed to load before the error, if any

    # CRITICAL FIX: Return exactly the (mols, names) tuple expected by app.py
    # If the tuple is not exactly size 2, app.py crashes.
    return mols, names


def ensure_3d_coords(mols: List[Chem.Mol]) -> List[Chem.Mol]:
    """
    Ensure each mol has 3D coordinates; generate via ETKDG if missing.
    """
    out = []
    for mol in mols:
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            try:
                # Add hydrogens for proper embedding
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                params.useRandomCoordinates = True  # Improved success rate
                result = AllChem.EmbedMolecule(mol, params)

                # Full Fallback method
                if result == -1:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

                # Perform Energy Minimization
                try:
                    if AllChem.MMFFHasAllMoleculeParams(mol):
                        AllChem.MMFFOptimizeMolecule(mol)
                    else:
                        AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    pass  # Continue even if energy minimization fails

                # Remove added Hydrogens unless needed by downstream stage
                mol = Chem.RemoveHs(mol)

            except Exception as e:
                print(
                    f"[DEBUG PREPROCESSING] 3D Coordinate generation error: {e}")

        out.append(mol)
    return out


# ── PDB active-site parsing ─────────────────────────────────────────────
def parse_pdb_binding_site(pdb_bytes: bytes) -> Dict[str, Any]:
    """
    Extract the centroid of the binding site from a PDB file.
    Heuristic: if HETATM records (non-water small molecules) are present,
    use their geometric centroid; otherwise fall back to all Cα atoms.

    Returns a dict with keys: centroid (x,y,z), residue_range, n_atoms, raw_text
    """
    try:
        if not pdb_bytes:
            return {
                "centroid": (
                    0.0,
                    0.0,
                    0.0),
                "n_atoms": 0,
                "source": "none",
                "raw_text": ""}

        text = pdb_bytes.decode("utf-8", errors="ignore")
        lines = text.splitlines()

        hetatm_coords: List[Tuple[float, float, float]] = []
        ca_coords: List[Tuple[float, float, float]] = []

        for line in lines:
            if len(line) < 54:  # Basic length check to avoid IndexErrors
                continue
            record = line[:6].strip()
            if record == "HETATM":
                res_name = line[17:20].strip()
                if res_name in ("HOH", "WAT", "DOD", "H2O"):
                    continue
                try:
                    x, y, z = float(line[30:38]), float(
                        line[38:46]), float(line[46:54])
                    hetatm_coords.append((x, y, z))
                except ValueError:
                    pass
            elif record == "ATOM":
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    try:
                        x, y, z = float(line[30:38]), float(
                            line[38:46]), float(line[46:54])
                        ca_coords.append((x, y, z))
                    except ValueError:
                        pass

        coords = hetatm_coords if hetatm_coords else ca_coords
        if not coords:
            return {"centroid": (0.0, 0.0, 0.0), "n_atoms": 0,
                    "source": "none", "raw_text": text[:500]}

        coords_array = np.array(coords)
        cx, cy, cz = np.mean(coords_array, axis=0)

        return {
            "centroid": (round(float(cx), 3), round(float(cy), 3), round(float(cz), 3)),
            "n_atoms": len(coords),
            "source": "HETATM" if hetatm_coords else "Cα centroid",
            "raw_text": text[:500],
        }

    except Exception as e:
        print(f"[DEBUG PREPROCESSING] Protein parsing error: {e}")
        # Graceful failure fallback
        return {
            "centroid": (
                0.0,
                0.0,
                0.0),
            "n_atoms": 0,
            "source": f"error: {
                str(e)}",
            "raw_text": ""}


# ── conf.txt grid parsing ───────────────────────────────────────────────
def parse_conf_txt(conf_bytes: bytes) -> Dict[str, Any]:
    """
    Parse an AutoDock-Vina style conf.txt for grid center and size.

    Expected format (any order, case-insensitive):
        center_x = 10.5
        center_y = -3.2
        center_z = 22.0
        size_x   = 20
        size_y   = 20
        size_z   = 20

    Returns a dict with keys: center (tuple), size (tuple), extra (dict)
    """
    kv: Dict[str, float] = {}

    if not conf_bytes:
        return {
            "center": (
                0.0, 0.0, 0.0), "size": (
                20.0, 20.0, 20.0), "extra": {}}

    try:
        text = conf_bytes.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip().lower().replace("-", "_")
                try:
                    # Robust handling for key/value parsing
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
            k: v for k,
            v in kv.items() if k not in (
                "center_x",
                "center_y",
                "center_z",
                "size_x",
                "size_y",
                "size_z")}

        return {"center": center, "size": size, "extra": extra}

    except Exception as e:
        print(f"[DEBUG PREPROCESSING] conf.txt parsing error: {e}")
        # Default fallback
        return {
            "center": (
                0.0, 0.0, 0.0), "size": (
                20.0, 20.0, 20.0), "extra": {
                "error": str(e)}}
