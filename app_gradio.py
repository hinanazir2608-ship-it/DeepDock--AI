"""
DeepDock-AI — app_gradio.py
Revised for:
    - Manual active-site/grid coordinates (no whole-protein centroid fallback)
    - Rigid-receptor GNINA docking
    - GNINA CNN scoring + CNN affinity
    - Fixed GNINA/RDKit seeds
    - Robust AltLoc handling
    - Full PDB-line preservation
    - Real/absent RMSD fields (no fake 0.0 values)
    - 3D preparation for BOTH CSV and SDF ligands
    - Robust MMFF -> UFF fallback
    - Ligand serial-number renumbering in complexes
    - Safe filenames
    - Generic Name column
    - Failed docking excluded from ADMET
    - Explicit ranking by CNN Score
    - Live Gradio progress updates
    - Temporary-run cleanup
    - share=False by default

IMPORTANT:
    This version intentionally requires the user to define the docking box.
    It does NOT calculate a whole-protein centroid and silently use it.

    For reproducibility:
      * GNINA_SEED and RDKIT_EMBED_SEED are fixed.
      * FORCE_CPU=False automatically uses an NVIDIA GPU when available;
      * FORCE_CPU=True forces CPU-only mode for maximum repeatability.
      * Exact bit-for-bit reproducibility still depends on the same
        GNINA/RDKit versions and computational environment.
"""

import os
import re
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem

from admetlab import run_admet_analysis


# ============================================================
# CONFIGURATION
# ============================================================

GNINA_SEED = 42
RDKIT_EMBED_SEED = 42

GNINA_NUM_MODES = 9
GNINA_EXHAUSTIVENESS = 8
GNINA_CPU = 4

# Automatic hardware selection:
# GPU is used when NVIDIA GPU is detected; otherwise GNINA falls back to CPU.
FORCE_CPU = False

# NVIDIA GPU device to use when GPU acceleration is available.
# 0 = first GPU (your environment currently exposes two Tesla T4 GPUs).
GNINA_GPU_DEVICE = 0

# Optional explicit GNINA executable path.
# Example:
# GNINA_PATH = r"C:\DeepDock-AI\gnina.exe"
GNINA_PATH = None

# Default manual active-site box.
DEFAULT_CENTER_X = 0.0
DEFAULT_CENTER_Y = 0.0
DEFAULT_CENTER_Z = 0.0

DEFAULT_SIZE_X = 20.0
DEFAULT_SIZE_Y = 20.0
DEFAULT_SIZE_Z = 20.0


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_file_path(f):
    """Works with older Gradio file objects and newer string paths."""
    if f is None:
        return None
    return getattr(f, "name", f)


def safe_filename(name):
    """Create a Windows-safe filename."""
    name = str(name).strip()
    if not name:
        name = "ligand"
    return re.sub(r"[^\w\-.]", "_", name)


def is_valid_number(value):
    """Accept legitimate 0.0 but reject None/NaN."""
    if value is None:
        return False
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_dataframe_indices(df):
    """
    Keep the identifier generic.
    Never relabel arbitrary molecule names as PubChem CID.
    """
    df = df.copy()
    df.index = range(1, len(df) + 1)
    return df


# ============================================================
# RECEPTOR PREPARATION
# ============================================================

def clean_receptor_pdb(pdb_file_path, output_path):
    """
    Prepare a protein-only receptor for rigid GNINA docking.

    Removes:
      - waters
      - HETATM records
      - alternate conformations B/C/etc.

    Keeps:
      - ATOM records
      - blank/A alternate location
      - original full PDB lines, including element columns

    Existing TER records are normalized so that duplicate TER lines
    are not produced.
    """
    kept = []

    with open(pdb_file_path, "r", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")

            if line.startswith("ATOM"):
                # AltLoc is column 17 (index 16).
                if len(line) > 16 and line[16] not in (" ", "A"):
                    continue

                # Preserve the COMPLETE line.
                kept.append(line)

            elif line.startswith(("ENDMDL", "MODEL")):
                # Ignore multi-model PDB bookkeeping.
                continue

            # Ignore original TER here; we add normalized TER below.

    if not kept:
        raise ValueError("No protein ATOM records were found in the target PDB.")

    # Insert TER between chains while avoiding duplicate TER records.
    output_lines = []
    previous_chain = None

    for line in kept:
        chain = line[21] if len(line) > 21 else " "

        if previous_chain is not None and chain != previous_chain:
            output_lines.append("TER")

        output_lines.append(line)
        previous_chain = chain

    if output_lines and output_lines[-1] != "TER":
        output_lines.append("TER")

    with open(output_path, "w", newline="\n") as f:
        for line in output_lines:
            f.write(line + "\n")

    return output_path


# ============================================================
# OPTIONAL NATIVE-LIGAND CENTER UTILITY
# ============================================================

def get_native_ligand_center(pdb_path, resname=None):
    """
    Optional utility for checking a native/co-crystal ligand center.

    This function is NOT used automatically by the docking pipeline,
    because this application is intentionally configured for manually
    supplied grid coordinates.
    """
    coords = []

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue

            if len(line) < 54:
                continue

            # Keep only blank/A alternate conformations.
            if len(line) > 16 and line[16] not in (" ", "A"):
                continue

            r = line[17:20].strip()

            if r in ("HOH", "WAT"):
                continue

            if resname and r != resname:
                continue

            try:
                coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue

    if not coords:
        return None

    return tuple(np.mean(np.asarray(coords), axis=0))


# ============================================================
# LIGAND 3D PREPARATION
# ============================================================

def prepare_ligand_3d(mol):
    """
    Generate/prepare a ligand for docking.

    Steps:
      1. Add explicit H atoms.
      2. Generate 3D coordinates with deterministic ETKDG.
      3. If embedding fails, retry using random coordinates.
      4. MMFF optimization.
      5. UFF fallback if MMFF is unavailable/fails.

    Returns:
        (prepared_mol, True, message)
    or
        (None, False, message)
    """
    if mol is None:
        return None, False, "RDKit returned None."

    try:
        mol = Chem.Mol(mol)

        # Make sure chemistry is sanitized as far as possible.
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Do not immediately discard molecules that can still be
            # processed by RDKit/GNINA.
            pass

        # Add explicit hydrogens.
        mol = Chem.AddHs(mol)

    except Exception as exc:
        return None, False, f"AddHs failed: {exc}"

    # If there is already a usable 3D conformer, keep it.
    has_3d = False
    try:
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            has_3d = conf.Is3D()
    except Exception:
        has_3d = False

    if not has_3d:
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = RDKIT_EMBED_SEED
            params.useRandomCoords = False

            conf_id = AllChem.EmbedMolecule(mol, params)

            # Correct fallback: random coordinates.
            if conf_id < 0:
                params = AllChem.ETKDGv3()
                params.randomSeed = RDKIT_EMBED_SEED
                params.useRandomCoords = True

                conf_id = AllChem.EmbedMolecule(mol, params)

            if conf_id < 0:
                return None, False, "3D embedding failed after deterministic and random-coordinate attempts."

        except Exception as exc:
            return None, False, f"3D embedding failed: {exc}"

    # MMFF returns -1 on failure; it does not necessarily raise.
    try:
        mmff_status = AllChem.MMFFOptimizeMolecule(mol)

        if mmff_status == -1:
            try:
                uff_status = AllChem.UFFOptimizeMolecule(mol)
                if uff_status == -1:
                    return mol, True, "3D generated; MMFF and UFF optimization did not converge."
                return mol, True, "3D generated; UFF fallback used."
            except Exception:
                return mol, True, "3D generated; MMFF failed and UFF fallback was unavailable."

    except Exception:
        try:
            uff_status = AllChem.UFFOptimizeMolecule(mol)
            if uff_status == -1:
                return mol, True, "3D generated; UFF did not converge."
            return mol, True, "3D generated; UFF fallback used."
        except Exception:
            return mol, True, "3D generated; geometry optimization unavailable."

    return mol, True, "3D generated and MMFF optimized."


# ============================================================
# LIGAND INPUT PROCESSING
# ============================================================

def find_smiles_column(df):
    for col in df.columns:
        if "smiles" in str(col).lower():
            return col
    return None


def find_name_column(df):
    preferred = ["name", "compound", "compound_name", "cid", "id"]

    lower_map = {str(c).lower(): c for c in df.columns}

    for key in preferred:
        if key in lower_map:
            return lower_map[key]

    return None


def process_ligands(ligand_file_path, filter_type="Lipinski"):
    """
    Read CSV or SDF and prepare every valid ligand for docking.

    Both CSV-derived and SDF-derived ligands receive 3D preparation.
    """
    ext = Path(ligand_file_path).suffix.lower()

    records = []
    prepared_mols = []
    rejected = []

    if ext in (".sdf", ".sd"):
        supplier = Chem.SDMolSupplier(
            ligand_file_path,
            removeHs=False,
            sanitize=True
        )

        for i, mol in enumerate(supplier, start=1):
            if mol is None:
                rejected.append({
                    "Name": f"SDF_{i}",
                    "Reason": "Invalid SDF molecule"
                })
                continue

            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"SDF_{i}"

            prepared, ok, prep_msg = prepare_ligand_3d(mol)

            if not ok:
                rejected.append({
                    "Name": name,
                    "Reason": prep_msg
                })
                continue

            try:
                mw = Descriptors.MolWt(prepared)
                logp = Descriptors.MolLogP(prepared)
                hbd = Lipinski.NumHDonors(prepared)
                hba = Lipinski.NumHAcceptors(prepared)

                passes = (
                    mw <= 500
                    and logp <= 5
                    and hbd <= 5
                    and hba <= 10
                )

                if filter_type.lower() == "lipinski" and not passes:
                    rejected.append({
                        "Name": name,
                        "Reason": "Failed Lipinski filter"
                    })
                    continue

                prepared.SetProp("_Name", name)

                records.append({
                    "Name": name,
                    "MW": round(mw, 2),
                    "LogP": round(logp, 2),
                    "HBD": int(hbd),
                    "HBA": int(hba),
                    "Preparation": prep_msg,
                })

                prepared_mols.append(prepared)

            except Exception as exc:
                rejected.append({
                    "Name": name,
                    "Reason": f"Descriptor calculation failed: {exc}"
                })

    elif ext in (".csv", ".txt"):
        df = pd.read_csv(ligand_file_path)

        smiles_col = find_smiles_column(df)
        if smiles_col is None:
            raise ValueError("CSV must contain a column whose name includes 'SMILES'.")

        name_col = find_name_column(df)

        for i, row in df.iterrows():
            smiles = str(row[smiles_col]).strip()

            if not smiles or smiles.lower() == "nan":
                rejected.append({
                    "Name": f"Mol_{i + 1}",
                    "Reason": "Missing SMILES"
                })
                continue

            name = (
                str(row[name_col]).strip()
                if name_col is not None and pd.notna(row[name_col])
                else f"Mol_{i + 1}"
            )

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                rejected.append({
                    "Name": name,
                    "Reason": "Invalid SMILES"
                })
                continue

            prepared, ok, prep_msg = prepare_ligand_3d(mol)

            if not ok:
                rejected.append({
                    "Name": name,
                    "Reason": prep_msg
                })
                continue

            try:
                mw = Descriptors.MolWt(prepared)
                logp = Descriptors.MolLogP(prepared)
                hbd = Lipinski.NumHDonors(prepared)
                hba = Lipinski.NumHAcceptors(prepared)

                passes = (
                    mw <= 500
                    and logp <= 5
                    and hbd <= 5
                    and hba <= 10
                )

                if filter_type.lower() == "lipinski" and not passes:
                    rejected.append({
                        "Name": name,
                        "Reason": "Failed Lipinski filter"
                    })
                    continue

                prepared.SetProp("_Name", name)

                records.append({
                    "Name": name,
                    "MW": round(mw, 2),
                    "LogP": round(logp, 2),
                    "HBD": int(hbd),
                    "HBA": int(hba),
                    "Preparation": prep_msg,
                })

                prepared_mols.append(prepared)

            except Exception as exc:
                rejected.append({
                    "Name": name,
                    "Reason": f"Descriptor calculation failed: {exc}"
                })

    else:
        raise ValueError("Supported ligand formats are CSV and SDF.")

    if not prepared_mols:
        raise ValueError("No ligands remained after input validation/filtering.")

    return (
        pd.DataFrame(records),
        prepared_mols,
        pd.DataFrame(rejected)
        if rejected
        else pd.DataFrame(columns=["Name", "Reason"])
    )


# ============================================================
# GNINA EXECUTABLE
# ============================================================

def get_gnina_path():
    """
    Locate GNINA executable.
    """
    candidates = []

    if GNINA_PATH:
        candidates.append(GNINA_PATH)

    for exe in ("gnina", "gninabase", "gnina.exe", "gninabase.exe"):
        found = shutil.which(exe)
        if found:
            candidates.append(found)

    # Search beside this application.
    app_dir = Path(__file__).resolve().parent

    for exe in ("gnina.exe", "gninabase.exe", "gnina", "gninabase"):
        candidates.append(str(app_dir / exe))

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)

        found = shutil.which(candidate) if candidate else None
        if found:
            return found

    raise FileNotFoundError(
        "GNINA executable was not found. Add gnina.exe to PATH or set GNINA_PATH."
    )


# ============================================================
# GNINA DOCKING
# ============================================================

def run_gnina_docking(
    gnina_exe,
    clean_target_path,
    ligand_path,
    output_path,
    center_x,
    center_y,
    center_z,
    size_x,
    size_y,
    size_z,
):
    """
    Rigid-receptor GNINA docking using a manually specified box.
    """
    cmd = [
        gnina_exe,
        "-r", clean_target_path,
        "-l", ligand_path,
        "-o", output_path,

        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),

        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),

        "--num_modes", str(GNINA_NUM_MODES),
        "--exhaustiveness", str(GNINA_EXHAUSTIVENESS),

        # CNN is used to rescore/rerank final poses.
        "--cnn_scoring", "rescore",

        # Reproducible docking seed.
        "--seed", str(GNINA_SEED),

            # CPU threads available to GNINA. This does NOT disable GPU acceleration.
        "--cpu", str(GNINA_CPU),
    ]

    # Explicitly select GPU when available; otherwise force CPU mode.
    gpu_available, _ = detect_gpu()
    use_gpu = gpu_available and not FORCE_CPU

    if use_gpu:
        cmd.extend(["--device", str(GNINA_GPU_DEVICE)])
    else:
        cmd.append("--no_gpu")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            errors="replace",
        )

        return result.stdout, result.stderr, result.returncode

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nGNINA docking timed out after 1800 seconds."
        return stdout, stderr, 124

    except Exception as exc:
        return "", str(exc), 1


# ============================================================
# GNINA SCORE PARSING
# ============================================================

def parse_gnina_scores(gnina_output_path):
    """
    Parse GNINA SDF properties.

    Keeps three scores separate:
      - Affinity (kcal/mol)
      - CNN Score
      - CNN Affinity

    No assumption is made that CNN Affinity has kcal/mol units.
    """
    supplier = Chem.SDMolSupplier(
        gnina_output_path,
        removeHs=False,
        sanitize=False
    )

    mols = [m for m in supplier if m is not None]

    if not mols:
        return None, None, None, None

    # GNINA's default pose sorting is CNNscore.
    # Therefore the first pose is the top CNN-score-ranked pose
    # unless --pose_sort_order is changed.
    top = mols[0]

    props = {p.strip(): top.GetProp(p) for p in top.GetPropNames()}

    normalized = {
        re.sub(r"[^a-z0-9]", "", k.lower()): v
        for k, v in props.items()
    }

    def get_exact(*keys):
        for key in keys:
            if key in normalized:
                try:
                    return float(normalized[key])
                except (TypeError, ValueError):
                    return None
        return None

    affinity = get_exact(
        "minimizedaffinity",
        "affinity",
    )

    cnn_score = get_exact(
        "cnnscore",
    )

    cnn_affinity = get_exact(
        "cnnaffinity",
    )

    return top, affinity, cnn_score, cnn_affinity


# ============================================================
# POSE / PDB / COMPLEX FUNCTIONS
# ============================================================

def write_top_pose_sdf(top_mol, output_path):
    writer = Chem.SDWriter(output_path)
    writer.write(top_mol)
    writer.close()
    return output_path


def ligand_mol_to_pdb(top_mol):
    """
    Convert the selected GNINA pose to a PDB block.
    """
    try:
        return Chem.MolToPDBBlock(top_mol)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert docked ligand to PDB: {exc}")


def renumber_ligand_pdb(ligand_pdb_block, starting_serial):
    """
    Convert ligand ATOM/HETATM records to HETATM and assign serials
    continuing after the receptor atom serial range.

    The ligand residue is written as LIG, chain Z, residue 1.
    """
    output = []
    serial = int(starting_serial)

    for raw in ligand_pdb_block.splitlines():
        line = raw.rstrip("\r\n")

        if not line.startswith(("ATOM", "HETATM")):
            continue

        serial += 1

        atom_name = line[12:16] if len(line) >= 16 else " C  "
        coords_tail = line[26:] if len(line) > 26 else ""

        # Preserve coordinate/occupancy/element information from the
        # original ligand PDB where available.
        new_line = (
            f"HETATM"
            f"{serial:5d}"
            f"{line[11:17] if len(line) >= 17 else atom_name}"
            f"LIG Z{1:4d} "
            f"{coords_tail}"
        )

        output.append(new_line)

    return output, serial


def create_complex_pdb(receptor_pdb_path, ligand_pdb_block, output_path):
    """
    Create a protein-ligand complex with non-conflicting atom serials.
    """
    with open(receptor_pdb_path, "r", errors="ignore") as f:
        rec_lines = [
            line.rstrip("\r\n")
            for line in f
            if line.startswith(("ATOM", "TER"))
        ]

    # Count protein ATOM records to establish a new serial range.
    n_rec = sum(1 for line in rec_lines if line.startswith("ATOM"))

    ligand_lines, _ = renumber_ligand_pdb(
        ligand_pdb_block,
        starting_serial=n_rec,
    )

    # Remove trailing TER/END duplicates from receptor input.
    while rec_lines and rec_lines[-1] == "TER":
        rec_lines.pop()

    with open(output_path, "w", newline="\n") as f:
        for line in rec_lines:
            f.write(line + "\n")

        f.write("TER\n")

        for line in ligand_lines:
            f.write(line + "\n")

        f.write("TER\n")
        f.write("END\n")

    return output_path


# ============================================================
# GPU CHECK
# ============================================================

def detect_gpu():
    """Return (available, names) for NVIDIA GPUs visible to this environment."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            errors="replace",
        )

        if result.returncode != 0:
            return False, []

        names = [
            line.strip()
            for line in (result.stdout or "").splitlines()
            if line.strip()
        ]
        return bool(names), names

    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False, []


# ============================================================
# ADMET
# ============================================================

def run_admet_for_successful_ligands(successful_mols, successful_names, score_map):
    """
    Run ADMET only for successfully docked ligands.

    The function keeps the existing admetlab.py interface assumption:
        run_admet_analysis(mols, names, scores)

    If your admetlab.py uses a different signature, only this small
    wrapper needs to be adjusted.
    """
    if not successful_mols:
        return pd.DataFrame()

    try:
        return run_admet_analysis(
            successful_mols,
            successful_names,
            score_map,
        )
    except TypeError:
        # Compatibility fallback for a common dataframe-based interface.
        rows = []

        for name, mol in zip(successful_names, successful_mols):
            rows.append({
                "Name": name,
                "SMILES": Chem.MolToSmiles(Chem.RemoveHs(mol)),
                "CNN Score": score_map.get(name),
            })

        admet_input = pd.DataFrame(rows)

        try:
            return run_admet_analysis(admet_input)
        except Exception as exc:
            return pd.DataFrame({
                "ADMET Status": [
                    f"ADMET execution failed: {exc}"
                ]
            })

    except Exception as exc:
        return pd.DataFrame({
            "ADMET Status": [
                f"ADMET execution failed: {exc}"
            ]
        })


# ============================================================
# MAIN PIPELINE
# ============================================================

def docking_pipeline(
    ligand_file,
    filter_type,
    target_file,
    use_custom_center,
    custom_cx,
    custom_cy,
    custom_cz,
    size_x,
    size_y,
    size_z,
):
    """
    Main DeepDock-AI workflow.

    The function is a generator so Gradio can display live progress.
    """
    status_log = "🚀 Starting DeepDock-AI...\n"

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    ligand_path = get_file_path(ligand_file)
    target_path = get_file_path(target_file)

    if not ligand_path:
        raise gr.Error("Please upload a ligand CSV or SDF file.")

    if not target_path:
        raise gr.Error("Please upload a target protein PDB file.")

    # --------------------------------------------------------
    # Validate manual active-site box
    # --------------------------------------------------------

    if not use_custom_center:
        raise gr.Error(
            "Manual active-site coordinates are required. "
            "Enable 'Use Manual Active-Site Coordinates'."
        )

    centers = [custom_cx, custom_cy, custom_cz]
    if not all(is_valid_number(x) for x in centers):
        raise gr.Error(
            "Center X, Center Y and Center Z must all contain numeric values."
        )

    sizes = [size_x, size_y, size_z]
    if not all(is_valid_number(x) and float(x) > 0 for x in sizes):
        raise gr.Error(
            "Grid Size X/Y/Z must all be positive numeric values."
        )

    cx, cy, cz = map(float, centers)
    sx, sy, sz = map(float, sizes)

    status_log += (
        "\n🎯 MANUAL ACTIVE-SITE BOX"
        f"\n   Center: ({cx:.3f}, {cy:.3f}, {cz:.3f}) Å"
        f"\n   Size:   ({sx:.1f}, {sy:.1f}, {sz:.1f}) Å"
        "\n   Receptor: rigid"
    )

    yield (
        status_log,
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        None,
        None,
    )

    # --------------------------------------------------------
    # Create run directory
    # --------------------------------------------------------

    run_dir = tempfile.mkdtemp(prefix="deepdock_ai_")

    try:
        clean_target = os.path.join(run_dir, "receptor_clean.pdb")
        docking_dir = os.path.join(run_dir, "docking")
        complexes_dir = os.path.join(run_dir, "complexes")

        os.makedirs(docking_dir, exist_ok=True)
        os.makedirs(complexes_dir, exist_ok=True)

        # ----------------------------------------------------
        # GNINA
        # ----------------------------------------------------

        try:
            gnina_exe = get_gnina_path()
        except FileNotFoundError as exc:
            raise gr.Error(str(exc))

        status_log += f"\n\n🔬 GNINA: {gnina_exe}"
        status_log += f"\n   Version: {get_gnina_version(gnina_exe)}"
        status_log += f"\n   Seed: {GNINA_SEED}"
        gpu_available, gpu_names = detect_gpu()
        use_gpu = gpu_available and not FORCE_CPU
        status_log += f"\n   CPU threads: {GNINA_CPU}"
        status_log += f"\n   Exhaustiveness: {GNINA_EXHAUSTIVENESS} | Modes: {GNINA_NUM_MODES}"
        status_log += f"\n   GPU detected: {'Yes' if gpu_available else 'No'}"
        status_log += f"\n   GPU(s): {', '.join(gpu_names) if gpu_names else 'None'}"
        status_log += f"\n   Hardware mode: {'GPU' if use_gpu else 'CPU'}"
        status_log += f"\n   GPU device: {GNINA_GPU_DEVICE if use_gpu else 'N/A'}"
        status_log += "\n   CNN scoring: rescore"
        status_log += "\n   Pose ranking: GNINA default CNN Score"

        yield (
            status_log,
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            None,
            None,
        )

        # ----------------------------------------------------
        # Receptor preparation
        # ----------------------------------------------------

        status_log += "\n\n🧬 Preparing receptor..."
        clean_receptor_pdb(target_path, clean_target)
        status_log += "\n   ✓ Waters/HETATM removed"
        status_log += "\n   ✓ AltLoc B+ removed; blank/A retained"
        status_log += "\n   ✓ Full PDB lines preserved"
        status_log += "\n   ✓ Rigid receptor prepared"

        # ----------------------------------------------------
        # Ligand preparation
        # ----------------------------------------------------

        status_log += "\n\n🧪 Preparing ligands..."

        ligand_df, prepared_mols, rejected_df = process_ligands(
            ligand_path,
            filter_type,
        )

        status_log += f"\n   ✓ Accepted ligands: {len(prepared_mols)}"
        status_log += f"\n   ✓ Rejected/skipped before docking: {len(rejected_df)}"

        yield (
            status_log,
            ligand_df,
            rejected_df,
            None,
            None,
            None,
        )

        # ----------------------------------------------------
        # Docking loop
        # ----------------------------------------------------

        docking_rows = []
        successful_mols = []
        successful_names = []
        score_map = {}

        total = len(prepared_mols)

        for idx, (mol, ligand_record) in enumerate(
            zip(prepared_mols, ligand_df.to_dict("records")),
            start=1,
        ):
            name = str(ligand_record["Name"])
            safe_name = safe_filename(name)

            status_log += (
                f"\n\n[{idx}/{total}] 🔄 Docking: {name}"
            )

            ligand_sdf = os.path.join(
                run_dir,
                f"{idx:03d}_{safe_name}.sdf"
            )

            output_sdf = os.path.join(
                docking_dir,
                f"{idx:03d}_{safe_name}_docked.sdf"
            )

            complex_pdb = os.path.join(
                complexes_dir,
                f"{idx:03d}_{safe_name}_complex.pdb"
            )

            try:
                writer = Chem.SDWriter(ligand_sdf)
                writer.write(mol)
                writer.close()

                stdout, stderr, returncode = run_gnina_docking(
                    gnina_exe=gnina_exe,
                    clean_target_path=clean_target,
                    ligand_path=ligand_sdf,
                    output_path=output_sdf,
                    center_x=cx,
                    center_y=cy,
                    center_z=cz,
                    size_x=sx,
                    size_y=sy,
                    size_z=sz,
                )

                if returncode != 0:
                    raise RuntimeError(
                        f"GNINA failed with exit code {returncode}. "
                        f"{stderr[-1000:]}"
                    )

                if not os.path.exists(output_sdf):
                    raise RuntimeError("GNINA finished but produced no output SDF.")

                (
                    top_mol,
                    affinity,
                    cnn_score,
                    cnn_affinity,
                ) = parse_gnina_scores(output_sdf)

                if top_mol is None:
                    raise RuntimeError("No valid docked pose was found.")

                if cnn_score is None:
                    raise RuntimeError("CNN Score was not found in GNINA output.")

                ligand_pdb = ligand_mol_to_pdb(top_mol)
                create_complex_pdb(
                    clean_target,
                    ligand_pdb,
                    complex_pdb,
                )

                docking_rows.append({
                    "Name": name,
                    "Affinity (kcal/mol)": (
                        round(affinity, 3)
                        if affinity is not None
                        else np.nan
                    ),
                    "CNN Score": round(cnn_score, 4),
                    "CNN Affinity": (
                        round(cnn_affinity, 4)
                        if cnn_affinity is not None
                        else np.nan
                    ),
                    "Docking Status": "Success",
                    "Complex PDB": complex_pdb,
                })

                successful_mols.append(mol)
                successful_names.append(name)
                score_map[name] = cnn_score

                status_log += (
                    "\n   ✓ Docking successful"
                    f"\n   Affinity: {affinity if affinity is not None else 'N/A'} kcal/mol"
                    f"\n   CNN Score: {cnn_score:.4f}"
                    f"\n   CNN Affinity: "
                    f"{cnn_affinity if cnn_affinity is not None else 'N/A'}"
                )

            except Exception as exc:
                docking_rows.append({
                    "Name": name,
                    "Affinity (kcal/mol)": np.nan,
                    "CNN Score": np.nan,
                    "CNN Affinity": np.nan,
                    "Docking Status": f"Failed: {exc}",
                    "Complex PDB": None,
                })

                status_log += f"\n   ✗ Failed: {exc}"

            current_df = pd.DataFrame(docking_rows)

            if not current_df.empty:
                # Higher CNN Score is better.
                current_df = current_df.sort_values(
                    by="CNN Score",
                    ascending=False,
                    na_position="last",
                ).reset_index(drop=True)

                current_df.insert(
                    0,
                    "Rank",
                    np.arange(1, len(current_df) + 1)
                )

            yield (
                status_log,
                ligand_df,
                current_df,
                None,
                None,
                None,
            )

        # ----------------------------------------------------
        # Final docking table
        # ----------------------------------------------------

        results_df = pd.DataFrame(docking_rows)

        if not results_df.empty:
            results_df = results_df.sort_values(
                by="CNN Score",
                ascending=False,
                na_position="last",
            ).reset_index(drop=True)

            results_df.insert(
                0,
                "Rank",
                np.arange(1, len(results_df) + 1)
            )

        # ----------------------------------------------------
        # ADMET — successful docking only
        # ----------------------------------------------------

        status_log += "\n\n🧬 Running ADMET only on successfully docked ligands..."

        admet_df = run_admet_for_successful_ligands(
            successful_mols,
            successful_names,
            score_map,
        )

        status_log += (
            f"\n   ✓ Successful docking ligands sent to ADMET: "
            f"{len(successful_mols)}"
        )

        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        docking_csv = os.path.join(
            run_dir,
            "DeepDockAI_Docking_Results.csv"
        )

        rejected_csv = os.path.join(
            run_dir,
            "DeepDockAI_Rejected_Ligands.csv"
        )

        admet_csv = os.path.join(
            run_dir,
            "DeepDockAI_ADMET_Results.csv"
        )

        results_df.to_csv(docking_csv, index=False)
        rejected_df.to_csv(rejected_csv, index=False)
        admet_df.to_csv(admet_csv, index=False)

        zip_path = os.path.join(
            run_dir,
            "DeepDockAI_Results.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as zf:

            zf.write(
                docking_csv,
                arcname=os.path.basename(docking_csv)
            )

            zf.write(
                rejected_csv,
                arcname=os.path.basename(rejected_csv)
            )

            zf.write(
                admet_csv,
                arcname=os.path.basename(admet_csv)
            )

            for root, _, files in os.walk(complexes_dir):
                for file in files:
                    path = os.path.join(root, file)
                    zf.write(
                        path,
                        arcname=os.path.join(
                            "complexes",
                            file
                        )
                    )

        status_log += "\n\n✅ DeepDock-AI completed."
        status_log += (
            "\n   Results ranked by CNN Score (descending)."
            "\n   Failed dockings were not sent to ADMET."
            "\n   RMSD l.b / RMSD u.b columns were intentionally removed."
        )

        yield (
            status_log,
            ligand_df,
            results_df,
            admet_df,
            docking_csv,
            zip_path,
        )

    finally:
        # The final files are returned by path. Gradio's file handling
        # copies/serves returned files; cleanup prevents accumulation.
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="DeepDock-AI — GNINA Rigid Docking"
) as demo:

    gr.Markdown(
        """
# 🧬 DeepDock-AI

### GNINA rigid-receptor docking with manual active-site coordinates

**Docking protocol**
- Manually defined active-site/grid center
- Manually defined grid dimensions
- Rigid receptor
- GNINA docking
- GNINA CNN rescoring
- Fixed random seed
- CPU mode enabled by default for stronger reproducibility

> **Important:** The application does not calculate a whole-protein centroid
> and does not silently perform blind docking. You must provide the active-site
> coordinates yourself.
"""
    )

    with gr.Row():

        ligand_file = gr.File(
            label="Ligands (.csv / .sdf)",
            file_types=[".csv", ".sdf", ".sd"],
        )

        target_file = gr.File(
            label="Target Protein (.pdb)",
            file_types=[".pdb"],
        )

    filter_type = gr.Dropdown(
        choices=["Lipinski", "None"],
        value="Lipinski",
        label="Ligand Filtering",
    )

    gr.Markdown("## 🎯 Manual Active-Site / Grid Definition")

    use_custom_center = gr.Checkbox(
        label="Use Manual Active-Site Coordinates",
        value=True,
    )

    with gr.Row():

        custom_cx = gr.Number(
            label="Center X (Å)",
            value=DEFAULT_CENTER_X,
        )

        custom_cy = gr.Number(
            label="Center Y (Å)",
            value=DEFAULT_CENTER_Y,
        )

        custom_cz = gr.Number(
            label="Center Z (Å)",
            value=DEFAULT_CENTER_Z,
        )

    with gr.Row():

        size_x = gr.Number(
            label="Grid Size X (Å)",
            value=DEFAULT_SIZE_X,
            minimum=1,
        )

        size_y = gr.Number(
            label="Grid Size Y (Å)",
            value=DEFAULT_SIZE_Y,
            minimum=1,
        )

        size_z = gr.Number(
            label="Grid Size Z (Å)",
            value=DEFAULT_SIZE_Z,
            minimum=1,
        )

    gr.Markdown(
        """
**For Mpro:** enter coordinates corresponding to the binding site you selected
(e.g., coordinates established from the co-crystal/native ligand or catalytic
site residues). Use the same box for all compounds in a comparative docking run.
"""
    )

    submit_btn = gr.Button(
        "🚀 Run GNINA Docking",
        variant="primary",
    )

    status_box = gr.Textbox(
        label="Live Progress",
        lines=18,
        interactive=False,
    )

    docking_table = gr.Dataframe(
        label="GNINA Docking Results",
        interactive=False,
    )

    rejected_table = gr.Dataframe(
        label="Rejected / Skipped Ligands",
        interactive=False,
    )

    admet_table = gr.Dataframe(
        label="ADMET Results",
        interactive=False,
    )

    docking_csv_file = gr.File(
        label="Download Docking CSV"
    )

    results_zip_file = gr.File(
        label="Download Complete Results ZIP"
    )

    submit_btn.click(
        fn=docking_pipeline,
        inputs=[
            ligand_file,
            filter_type,
            target_file,
            use_custom_center,
            custom_cx,
            custom_cy,
            custom_cz,
            size_x,
            size_y,
            size_z,
        ],
        outputs=[
            status_box,
            docking_table,
            rejected_table,
            admet_table,
            docking_csv_file,
            results_zip_file,
        ],
    )


if __name__ == "__main__":
    # share=False intentionally prevents creating a public tunnel.
    demo.launch(
        share=False
    )
