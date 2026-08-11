"""
DeepDock-AI — docking.py (fixed)

Changes vs. original:
  1. Fixed RDKit ETKDG embedding seed -> same starting 3D geometry every run/platform.
  2. Fixed Vina --seed and --cpu -> reproducible search across machines.
  3. Reads the score from the --out PDBQT's "REMARK VINA RESULT:" line instead of
     scraping stdout. This is far more stable across Vina 1.1.x / 1.2.x CLI versions
     than parsing the printed results table.
  4. Removed the silent fake-score fallback. Failures are now reported honestly
     (Status = "Failed: <reason>", Affinity = None) instead of a linear fake number
     labeled "Docked Successfully".
  5. Validates grid_center isn't left at the (0,0,0) default before docking.
  6. Checks the Vina binary version once up front and logs it, so you can diff
     Kaggle vs. Ubuntu builds directly instead of guessing.
"""

import os
import shutil
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

RDKIT_EMBED_SEED = 42
VINA_SEED = 42
VINA_CPU = 4  # pin the SAME value on every platform you compare


def get_vina_version():
    """Returns the Vina binary's reported version string, or None if not found."""
    vina_path = shutil.which("vina")
    if not vina_path:
        return None
    try:
        result = subprocess.run(["vina", "--version"], capture_output=True, text=True, timeout=10)
        return (result.stdout or result.stderr).strip()
    except Exception:
        return None


def mol_to_pdbqt(mol, output_pdbqt_path):
    """
    Converts an RDKit Mol object directly into a valid PDBQT file format 
    using RDKit's built-in PDB block generation.
    Returns (success: bool, error_message: str | None).
    """
    try:
        # Directory ensure karein
        os.makedirs(os.path.dirname(output_pdbqt_path), exist_ok=True)

        # Hydrogens add karein aur 3D coordinates generate karein agar missing hain
        mol = Chem.AddHs(mol)
        if not mol.GetNumConformers():
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) == -1:
                return False, "RDKit 3D embedding failed."
            AllChem.MMFFOptimizeMolecule(mol)

        # RDKit se PDB block nikal kar usay Vina-compatible PDBQT lines mein convert karein
        pdb_block = Chem.MolToPDBBlock(mol)
        
        pdbqt_lines = []
        for line in pdb_block.splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Vina ke liye default atom type aur charges append kar dein
                atom_line = line[:66] + "     0.00  0.00    C\n"
                pdbqt_lines.append(atom_line)
            elif line.startswith("CONECT") or line.startswith("ROOT") or line.startswith("ENDROOT"):
                pdbqt_lines.append(line + "\n")
                
        pdbqt_lines.append("END\n")

        with open(output_pdbqt_path, "w") as f:
            f.writelines(pdbqt_lines)

        if os.path.exists(output_pdbqt_path) and os.path.getsize(output_pdbqt_path) > 0:
            return True, None
        else:
            return False, "Generated PDBQT file is empty."

    except Exception as e:
        return False, f"Error generating PDBQT: {e}"


def parse_affinity_from_pdbqt(out_pdbqt_path):
    """
    Reads the top-mode binding affinity from Vina's output PDBQT file, e.g.:
        REMARK VINA RESULT:    -7.6      0.000      0.000
    This line format has been far more stable across Vina 1.1.x/1.2.x than the
    printed stdout table, since it's part of Vina's file-output contract.
    """
    if not os.path.exists(out_pdbqt_path):
        return None
    with open(out_pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                # ["REMARK", "VINA", "RESULT:", "-7.6", "0.000", "0.000"]
                try:
                    return float(parts[3])
                except (IndexError, ValueError):
                    return None
    return None


def run_batched_docking(
    filtered_mols,
    target_pdbqt_file,
    grid_center=(0, 0, 0),
    grid_size=(20, 20, 20),
    output_dir="/kaggle/working/output",
    progress=None,
):
    """
    Executes AutoDock Vina docking with live progress tracking and honest
    failure reporting (no fabricated scores).
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(filtered_mols)
    results = []

    target_path = target_pdbqt_file.name if hasattr(target_pdbqt_file, "name") else str(target_pdbqt_file)

    vina_version = get_vina_version()
    if vina_version is None:
        print("⚠️  'vina' binary not found on PATH. All molecules will fail — "
              "install it before docking, don't rely on a fallback score.")
    else:
        print(f"ℹ️  Using Vina: {vina_version}")

    if grid_center == (0, 0, 0):
        print("⚠️  grid_center is still the default (0,0,0). This is almost never "
              "your real binding site — pass the actual pocket coordinates.")

    for idx, mol in enumerate(filtered_mols):
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Ligand_{idx+1}"

        if progress is not None:
            current_progress = 0.2 + (0.6 * (idx + 1) / total)
            progress(current_progress, desc=f"⚡ Docking [{idx+1}/{total}]: {mol_name}")

        ligand_pdbqt = os.path.join(output_dir, f"{mol_name}.pdbqt")
        out_pdbqt = os.path.join(output_dir, f"{mol_name}_out.pdbqt")

        affinity = None
        status = None

        converted, convert_err = mol_to_pdbqt(mol, ligand_pdbqt)

        if not converted:
            status = f"Failed: {convert_err}"
        elif not os.path.exists(target_path):
            status = f"Failed: receptor PDBQT not found at {target_path}"
        elif vina_version is None:
            status = "Failed: vina binary not found on PATH"
        else:
            vina_cmd = [
                "vina",
                "--receptor", target_path,
                "--ligand", ligand_pdbqt,
                "--center_x", str(grid_center[0]),
                "--center_y", str(grid_center[1]),
                "--center_z", str(grid_center[2]),
                "--size_x", str(grid_size[0]),
                "--size_y", str(grid_size[1]),
                "--size_z", str(grid_size[2]),
                "--out", out_pdbqt,
                "--exhaustiveness", "8",
                "--seed", str(VINA_SEED),
                "--cpu", str(VINA_CPU),
            ]
            process = subprocess.run(vina_cmd, capture_output=True, text=True)

            if process.returncode != 0:
                status = f"Failed: vina exited {process.returncode}: {process.stderr.strip()[:300]}"
            else:
                affinity = parse_affinity_from_pdbqt(out_pdbqt)
                status = "Docked Successfully" if affinity is not None else \
                    "Failed: vina ran but no REMARK VINA RESULT line found in output"

        results.append({
            "CID": idx + 1,
            "Molecule Name": mol_name,
            "Affinity (kcal/mol)": affinity,
            "Status": status,
        })

    return results
