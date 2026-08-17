import os
import zipfile
import tempfile
import pandas as pd
from rdkit import Chem

def pdbqt_to_pdb_standard(pdbqt_file: str, output_pdb_path: str = None) -> str:
    """
    Converts a PDBQT docking output or target file into standard PDB format by stripping 
    AutoDock charge and atom-type columns, and ensuring proper formatting for downstream 
    visualizers like Discovery Studio.
    """
    if not os.path.exists(pdbqt_file):
        raise FileNotFoundError(f"PDBQT file not found: {pdbqt_file}")

    if output_pdb_path is None:
        output_pdb_path = pdbqt_file.replace(".pdbqt", ".pdb")

    pdb_lines = []
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # Retain standard PDB columns (1 to 66 characters) to remove AutoDock charges/types
                standard_line = line[:66].ljust(66) + "\n"
                pdb_lines.append(standard_line)
            elif line.startswith(("MODEL", "ENDMDL", "TER", "HEADER", "TITLE", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR", "REMARK")):
                pdb_lines.append(line)

    with open(output_pdb_path, "w") as f:
        f.writelines(pdb_lines)

    return output_pdb_path


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
