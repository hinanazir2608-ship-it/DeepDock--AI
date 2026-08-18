import os
import zipfile
import tempfile
import pandas as pd
from rdkit import Chem

# AutoDock/Vina PDBQT atom types aren't real element symbols — map them back.
AD_TYPE_TO_ELEMENT = {
    "H": "H", "HD": "H", "HS": "H",
    "C": "C", "A": "C",              # A = aromatic carbon
    "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O",
    "S": "S", "SA": "S",
    "P": "P", "F": "F",
    "Cl": "Cl", "CL": "Cl",
    "Br": "Br", "BR": "Br",
    "I": "I",
    "Mg": "Mg", "MG": "Mg", "Ca": "Ca", "CA": "Ca",
    "Mn": "Mn", "MN": "Mn", "Fe": "Fe", "FE": "Fe",
    "Zn": "Zn", "ZN": "Zn", "Cu": "Cu", "CU": "Cu",
    "Na": "Na", "K": "K",
}

def _guess_element(line: str) -> str:
    """Real element for a PDBQT atom line: try the AutoDock type (last
    whitespace token), fall back to the atom name if unmapped."""
    tokens = line.split()
    ad_type = tokens[-1] if tokens else ""
    if ad_type in AD_TYPE_TO_ELEMENT:
        return AD_TYPE_TO_ELEMENT[ad_type]
    name = line[12:16].strip()
    if len(name) >= 2 and name[:2].capitalize() in ("Cl", "Br", "Zn", "Fe", "Mg", "Mn", "Ca", "Na"):
        return name[:2].capitalize()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return " "


def pdbqt_to_pdb_standard(pdbqt_file: str, output_pdb_path: str = None, start_serial: int = 1) -> tuple:
    """
    Converts a PDBQT (receptor or docked ligand) into standard PDB format.

    Fixes:
    - Writes the REAL element symbol (cols 77-78), mapped from the AutoDock
      atom type, instead of silently dropping it.
    - Renumbers atom serials starting at `start_serial`, so receptor + ligand
      can be concatenated with zero collisions.

    Returns (output_pdb_path, next_free_serial) — chain receptor then ligand:
        receptor_pdb, next_serial = pdbqt_to_pdb_standard(receptor_pdbqt, start_serial=1)
        ligand_pdb, _ = pdbqt_to_pdb_standard(ligand_pdbqt, start_serial=next_serial)
    """
    if not os.path.exists(pdbqt_file):
        raise FileNotFoundError(f"PDBQT file not found: {pdbqt_file}")
    if output_pdb_path is None:
        output_pdb_path = pdbqt_file.replace(".pdbqt", ".pdb")

    pdb_lines = []
    serial = start_serial

    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                element = _guess_element(line)
                base = line[:66].ljust(66)
                renumbered = base[:6] + f"{serial:>5}" + base[11:]   # swap serial only
                standard_line = renumbered + " " * 10 + f"{element:>2}" + "  " + "\n"
                pdb_lines.append(standard_line)
                serial += 1
            elif line.startswith(("MODEL", "ENDMDL", "TER", "HEADER", "TITLE", "COMPND",
                                   "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR", "REMARK")):
                pdb_lines.append(line)

    with open(output_pdb_path, "w") as f:
        f.writelines(pdb_lines)

    return output_pdb_path, serial  # `serial` is now the next free number

def create_results_zip(zip_output_path: str, csv_path: str, docked_mols: list = None, mol_names: list = None) -> str:
    """
    Packages screening CSV reports and docked structures into a downloadable ZIP archive.
    """
    with zipfile.ZipFile(zip_output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        # Add CSV Report if available
        if csv_path and os.path.exists(csv_path):
            zipf.write(csv_path, arcname=os.path.basename(csv_path))

        # Add Docked Molecular Structures if provided
        if docked_mols:
            dock_dir = tempfile.mkdtemp(prefix="docked_complexes_")

            for idx, mol in enumerate(docked_mols):
                name = mol_names[idx] if (mol_names and idx < len(mol_names)) else f"compound_{idx+1}"
                safe_name = f"{idx}_{name}"
                file_name = os.path.join(dock_dir, f"{safe_name}_docked.sdf")

                try:
                    if isinstance(mol, Chem.Mol):
                        writer = Chem.SDWriter(file_name)
                        writer.write(mol)
                        writer.close()
                        if os.path.exists(file_name):
                            zipf.write(file_name, arcname=f"docked_structures/{safe_name}_docked.sdf")
                except Exception as e:
                    print(f"Warning: Could not write structure for {name}: {str(e)}")

    print(f"📦 Created results ZIP at: {zip_output_path}")
    return zip_output_path
