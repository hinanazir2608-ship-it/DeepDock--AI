"""
DeepDock-AI — docking.py

Reproducible AutoDock Vina docking module.

Reproducibility measures:
  1. Fixed RDKit ETKDG seed for ligand 3D embedding.
  2. Fixed Vina random seed.
  3. Fixed Vina CPU count.
  4. Existing ligand conformers are replaced with a deterministic conformer.
  5. Ligand hydrogens are added before 3D preparation.
  6. Vina affinity is read from the output PDBQT REMARK VINA RESULT line.
  7. No fabricated/fallback docking scores.
  8. Vina executable is detected from PATH rather than hard-coded to
     /usr/local/bin/vina.
  9. Vina version is recorded before docking.
 10. Grid center and size are validated.
 11. Each ligand gets an independent deterministic Vina run.
"""

import os
import shutil
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================
# REPRODUCIBILITY SETTINGS
# ============================================================

RDKIT_EMBED_SEED = 42
VINA_SEED = 42

# Keep this identical on every machine being compared.
VINA_CPU = 4

# Vina search settings
VINA_EXHAUSTIVENESS = 8

# Default grid dimensions
DEFAULT_GRID_SIZE = (20.0, 20.0, 20.0)


# ============================================================
# VINA EXECUTABLE
# ============================================================

def get_vina_path():
    """
    Locate AutoDock Vina from the system PATH.

    This avoids hard-coding /usr/local/bin/vina, which may differ
    between Kaggle, Ubuntu, Windows, Docker, etc.
    """
    vina_path = shutil.which("vina")

    if vina_path:
        return vina_path

    return None


def get_vina_version(vina_path=None):
    """
    Return the Vina version string.
    """

    if vina_path is None:
        vina_path = get_vina_path()

    if vina_path is None:
        return None

    try:
        result = subprocess.run(
            [vina_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        version = (result.stdout or result.stderr).strip()

        return version if version else None

    except Exception:
        return None


# ============================================================
# LIGAND 3D PREPARATION
# ============================================================

def prepare_deterministic_3d_molecule(mol):
    """
    Create a deterministic 3D ligand conformer.

    A fresh conformer is generated for every docking run using
    a fixed ETKDG seed.

    Returns:
        (prepared_mol, error_message)
    """

    try:
        if mol is None:
            return None, "Invalid RDKit molecule."

        # Work on a copy so the original molecule is not modified.
        mol = Chem.Mol(mol)

        # Add explicit hydrogens.
        mol = Chem.AddHs(mol)

        # Remove any existing conformers.
        mol.RemoveAllConformers()

        # Deterministic ETKDG embedding.
        params = AllChem.ETKDGv3()
        params.randomSeed = RDKIT_EMBED_SEED

        # Keep the embedding deterministic.
        params.useRandomCoords = False

        result = AllChem.EmbedMolecule(
            mol,
            params,
        )

        if result == -1:
            return None, "RDKit 3D embedding failed."

        # Deterministic MMFF optimization when possible.
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(
                    mol,
                    maxIters=200,
                )
            else:
                # UFF fallback if MMFF parameters are unavailable.
                AllChem.UFFOptimizeMolecule(
                    mol,
                    maxIters=200,
                )
        except Exception:
            # Keep the embedded coordinates if optimization fails.
            pass

        return mol, None

    except Exception as e:
        return None, f"3D preparation error: {e}"


# ============================================================
# PDB → PDBQT LIGAND CONVERSION
# ============================================================

def mol_to_pdbqt(mol, output_pdbqt_path):
    """
    Convert an RDKit molecule to a Vina-compatible PDBQT file.

    The molecule is regenerated deterministically so that the same
    input SMILES produces the same starting 3D coordinates when the
    same RDKit version/environment is used.
    """

    try:
        output_dir = os.path.dirname(output_pdbqt_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        prepared_mol, error = prepare_deterministic_3d_molecule(mol)

        if prepared_mol is None:
            return False, error

        pdb_block = Chem.MolToPDBBlock(prepared_mol)

        pdbqt_lines = []
        atom_serial = 1

        for line in pdb_block.splitlines():

            if not (
                line.startswith("ATOM")
                or line.startswith("HETATM")
            ):
                continue

            atom_name = line[12:16].strip()

            # Determine element from RDKit atom information whenever
            # possible rather than assuming atom_name[0].
            try:
                atom = prepared_mol.GetAtomWithIdx(atom_serial - 1)
                element = atom.GetSymbol()
            except Exception:
                element = atom_name[0] if atom_name else "C"

            # Coordinates from the original PDB line.
            coordinates = line[30:54]

            new_line = (
                f"ATOM  "
                f"{atom_serial:5d} "
                f"{atom_name:<4s} "
                f"UNL     "
                f"1    "
                f"{coordinates}"
                f"  1.00  0.00    "
                f"+0.000 "
                f"{element:>2s}\n"
            )

            pdbqt_lines.append(new_line)
            atom_serial += 1

        if not pdbqt_lines:
            return False, "No valid atoms found for PDBQT conversion."

        with open(
            output_pdbqt_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines(pdbqt_lines)

        return True, None

    except Exception as e:
        return False, f"Error generating PDBQT: {e}"


# ============================================================
# AFFINITY EXTRACTION
# ============================================================

def parse_affinity_from_pdbqt(out_pdbqt_path):
    """
    Extract the top Vina affinity from the output PDBQT.

    Expected line:

        REMARK VINA RESULT:    -7.6      0.000      0.000
    """

    if not os.path.exists(out_pdbqt_path):
        return None

    try:
        with open(
            out_pdbqt_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            for line in f:

                if line.startswith("REMARK VINA RESULT:"):

                    parts = line.split()

                    # Example:
                    # ["REMARK", "VINA", "RESULT:", "-7.6", "0.000", "0.000"]

                    if len(parts) >= 4:
                        try:
                            return float(parts[3])
                        except ValueError:
                            return None

        return None

    except Exception:
        return None


# ============================================================
# GRID VALIDATION
# ============================================================

def validate_grid(grid_center, grid_size):
    """
    Validate docking grid parameters.
    """

    if grid_center is None or len(grid_center) != 3:
        return False, "grid_center must contain exactly 3 coordinates."

    if grid_size is None or len(grid_size) != 3:
        return False, "grid_size must contain exactly 3 dimensions."

    try:
        center = tuple(float(x) for x in grid_center)
        size = tuple(float(x) for x in grid_size)
    except Exception:
        return False, "Grid coordinates and dimensions must be numeric."

    if any(x <= 0 for x in size):
        return False, "Grid dimensions must be greater than zero."

    # Prevent accidental default docking at the origin.
    if center == (0.0, 0.0, 0.0):
        return False, (
            "grid_center is (0,0,0). "
            "Provide the actual binding-site coordinates."
        )

    return True, None


# ============================================================
# SINGLE LIGAND DOCKING
# ============================================================

def dock_single_ligand(
    mol,
    target_pdbqt_file,
    grid_center,
    grid_size,
    output_dir,
    vina_path,
):
    """
    Dock one ligand using deterministic settings.

    Returns:
        affinity, status, ligand_pdbqt, output_pdbqt
    """

    mol_name = (
        mol.GetProp("_Name")
        if mol.HasProp("_Name")
        else "Ligand"
    )

    # Make filesystem-safe filename.
    safe_name = "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in mol_name
    )

    ligand_pdbqt = os.path.join(
        output_dir,
        f"{safe_name}.pdbqt",
    )

    output_pdbqt = os.path.join(
        output_dir,
        f"{safe_name}_out.pdbqt",
    )

    # Remove previous output so an old result cannot accidentally
    # be interpreted as the result of the current run.
    for path in (ligand_pdbqt, output_pdbqt):

        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # --------------------------------------------------------
    # Prepare ligand
    # --------------------------------------------------------

    converted, error = mol_to_pdbqt(
        mol,
        ligand_pdbqt,
    )

    if not converted:
        return (
            None,
            f"Failed: {error}",
            ligand_pdbqt,
            output_pdbqt,
        )

    # --------------------------------------------------------
    # Validate receptor
    # --------------------------------------------------------

    if not os.path.exists(target_pdbqt_file):
        return (
            None,
            f"Failed: receptor PDBQT not found: {target_pdbqt_file}",
            ligand_pdbqt,
            output_pdbqt,
        )

    # --------------------------------------------------------
    # Vina command
    # --------------------------------------------------------

    vina_cmd = [
        vina_path,

        "--receptor",
        target_pdbqt_file,

        "--ligand",
        ligand_pdbqt,

        "--center_x",
        str(grid_center[0]),

        "--center_y",
        str(grid_center[1]),

        "--center_z",
        str(grid_center[2]),

        "--size_x",
        str(grid_size[0]),

        "--size_y",
        str(grid_size[1]),

        "--size_z",
        str(grid_size[2]),

        "--out",
        output_pdbqt,

        "--exhaustiveness",
        str(VINA_EXHAUSTIVENESS),

        "--seed",
        str(VINA_SEED),

        "--cpu",
        str(VINA_CPU),
    ]

    # --------------------------------------------------------
    # Run Vina
    # --------------------------------------------------------

    try:

        process = subprocess.run(
            vina_cmd,
            capture_output=True,
            text=True,
        )

    except Exception as e:

        return (
            None,
            f"Failed: could not execute Vina: {e}",
            ligand_pdbqt,
            output_pdbqt,
        )

    # --------------------------------------------------------
    # Check Vina exit status
    # --------------------------------------------------------

    if process.returncode != 0:

        stderr = (
            process.stderr.strip()
            if process.stderr
            else "Unknown Vina error."
        )

        return (
            None,
            f"Failed: Vina exited {process.returncode}: "
            f"{stderr[:500]}",
            ligand_pdbqt,
            output_pdbqt,
        )

    # --------------------------------------------------------
    # Extract affinity from output PDBQT
    # --------------------------------------------------------

    affinity = parse_affinity_from_pdbqt(
        output_pdbqt
    )

    if affinity is None:

        return (
            None,
            "Failed: Vina completed but no "
            "REMARK VINA RESULT was found.",
            ligand_pdbqt,
            output_pdbqt,
        )

    return (
        affinity,
        "Docked Successfully",
        ligand_pdbqt,
        output_pdbqt,
    )


# ============================================================
# BATCH DOCKING
# ============================================================

def run_batched_docking(
    filtered_mols,
    target_pdbqt_file,
    grid_center=(0, 0, 0),
    grid_size=DEFAULT_GRID_SIZE,
    output_dir="/kaggle/working/output",
    progress=None,
):
    """
    Execute reproducible AutoDock Vina docking.

    Important:
      - Same ligand preparation seed
      - Same Vina seed
      - Same CPU count
      - Same exhaustiveness
      - Same grid
      - Same Vina version

    Therefore, results are reproducible when the same software
    versions and input files are used.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    results = []

    total = len(filtered_mols)

    if total == 0:
        return results

    # --------------------------------------------------------
    # Locate Vina
    # --------------------------------------------------------

    vina_path = get_vina_path()

    if vina_path is None:

        print(
            "❌ AutoDock Vina was not found on PATH. "
            "Install Vina before docking."
        )

        for idx, mol in enumerate(filtered_mols):

            mol_name = (
                mol.GetProp("_Name")
                if mol.HasProp("_Name")
                else f"Ligand_{idx + 1}"
            )

            results.append({
                "CID": idx + 1,
                "Molecule Name": mol_name,
                "Affinity (kcal/mol)": None,
                "Status": "Failed: Vina binary not found on PATH",
            })

        return results

    # --------------------------------------------------------
    # Vina version
    # --------------------------------------------------------

    vina_version = get_vina_version(vina_path)

    print(f"ℹ️ Vina executable: {vina_path}")
    print(
        f"ℹ️ Vina version: "
        f"{vina_version if vina_version else 'Unknown'}"
    )

    print(f"ℹ️ Vina seed: {VINA_SEED}")
    print(f"ℹ️ Vina CPU: {VINA_CPU}")
    print(f"ℹ️ Vina exhaustiveness: {VINA_EXHAUSTIVENESS}")
    print(f"ℹ️ RDKit embedding seed: {RDKIT_EMBED_SEED}")

    # --------------------------------------------------------
    # Validate receptor
    # --------------------------------------------------------

    target_path = (
        target_pdbqt_file.name
        if hasattr(target_pdbqt_file, "name")
        else str(target_pdbqt_file)
    )

    if not os.path.exists(target_path):

        print(
            f"❌ Receptor PDBQT not found: {target_path}"
        )

        for idx, mol in enumerate(filtered_mols):

            mol_name = (
                mol.GetProp("_Name")
                if mol.HasProp("_Name")
                else f"Ligand_{idx + 1}"
            )

            results.append({
                "CID": idx + 1,
                "Molecule Name": mol_name,
                "Affinity (kcal/mol)": None,
                "Status": (
                    f"Failed: receptor PDBQT not found "
                    f"at {target_path}"
                ),
            })

        return results

    # --------------------------------------------------------
    # Validate grid
    # --------------------------------------------------------

    grid_ok, grid_error = validate_grid(
        grid_center,
        grid_size,
    )

    if not grid_ok:

        print(f"❌ Invalid docking grid: {grid_error}")

        for idx, mol in enumerate(filtered_mols):

            mol_name = (
                mol.GetProp("_Name")
                if mol.HasProp("_Name")
                else f"Ligand_{idx + 1}"
            )

            results.append({
                "CID": idx + 1,
                "Molecule Name": mol_name,
                "Affinity (kcal/mol)": None,
                "Status": f"Failed: {grid_error}",
            })

        return results

    print(
        "ℹ️ Grid center: "
        f"{grid_center[0]}, "
        f"{grid_center[1]}, "
        f"{grid_center[2]}"
    )

    print(
        "ℹ️ Grid size: "
        f"{grid_size[0]}, "
        f"{grid_size[1]}, "
        f"{grid_size[2]}"
    )

    # --------------------------------------------------------
    # Dock each ligand
    # --------------------------------------------------------

    for idx, mol in enumerate(filtered_mols):

        mol_name = (
            mol.GetProp("_Name")
            if mol.HasProp("_Name")
            else f"Ligand_{idx + 1}"
        )

        if progress is not None:

            current_progress = (
                0.2
                + (
                    0.6
                    * (idx + 1)
                    / total
                )
            )

            progress(
                current_progress,
                desc=(
                    f"⚡ Docking "
                    f"[{idx + 1}/{total}]: "
                    f"{mol_name}"
                ),
            )

        print(
            f"\n🔬 Docking "
            f"[{idx + 1}/{total}]: "
            f"{mol_name}"
        )

        affinity, status, ligand_pdbqt, output_pdbqt = (
            dock_single_ligand(
                mol=mol,
                target_pdbqt_file=target_path,
                grid_center=grid_center,
                grid_size=grid_size,
                output_dir=output_dir,
                vina_path=vina_path,
            )
        )

        results.append({
            "CID": idx + 1,
            "Molecule Name": mol_name,
            "Affinity (kcal/mol)": affinity,
            "Status": status,
        })

        if affinity is not None:
            print(
                f"   ✔ Affinity: "
                f"{affinity:.3f} kcal/mol"
            )
        else:
            print(
                f"   ⚠️ {status}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    successful = sum(
        1
        for result in results
        if result["Affinity (kcal/mol)"] is not None
    )

    failed = total - successful

    print("\n======================================")
    print("DOCKING COMPLETE")
    print("======================================")
    print(f"Total ligands: {total}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")
    print(f"Vina version:   {vina_version}")
    print(f"Vina seed:      {VINA_SEED}")
    print(f"Vina CPU:       {VINA_CPU}")
    print(f"Exhaustiveness: {VINA_EXHAUSTIVENESS}")
    print("======================================")

    return results
