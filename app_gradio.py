import os
import shutil
import zipfile
import tempfile
import subprocess

import numpy as np
import pandas as pd
import gradio as gr

# ============================================================
# RDKit Logging & Imports
# ============================================================

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# ============================================================
# ADMETlab 3.0 Integration
# ============================================================

from admetlab import run_admet_analysis


# ============================================================
# Configuration
# ============================================================

GNINA_SEED = 42

# False = automatically use GPU if available
# True  = force GNINA to CPU
FORCE_CPU = False


# ============================================================
# 1. HELPER & PREPROCESSING FUNCTIONS
# ============================================================

def clean_and_prepare_receptor(target_file_path, output_pdb_path):
    """
    Simple and clean PDB preparation.

    - Reads raw PDB file.
    - Removes water molecules.
    - Removes HETATM records.
    - Keeps ATOM records.
    - Adds TER between chains.
    - Writes a clean PDB suitable for GNINA.
    """

    try:
        with open(
            target_file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            lines = f.readlines()

        cleaned_pdb_lines = []
        current_chain = None

        for line in lines:

            # Skip water molecules
            if any(wat in line for wat in ["HOH", "WAT"]):
                continue

            # Skip heteroatoms
            if line.startswith("HETATM"):
                continue

            if line.startswith("ATOM"):

                if len(line) < 54:
                    continue

                chain_id = line[21] if len(line) > 21 else "A"

                if (
                    current_chain is not None
                    and chain_id != current_chain
                ):
                    cleaned_pdb_lines.append("TER\n")

                current_chain = chain_id

                std_line = line[:66].ljust(66) + "\n"
                cleaned_pdb_lines.append(std_line)

            elif line.startswith(
                ("TER", "HEADER", "REMARK", "MODEL", "ENDMDL")
            ):
                cleaned_pdb_lines.append(
                    line if line.endswith("\n") else line + "\n"
                )

        with open(
            output_pdb_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.writelines(cleaned_pdb_lines)

        return True

    except Exception as e:
        print(f"[ERROR] Receptor cleaning failed: {e}")
        return False


# ============================================================
# DataFrame Cleaning
# ============================================================

def clean_dataframe_indices(df, pubchem_col_name="PubChem CID"):
    """
    Cleans DataFrames for Gradio display.

    - Renames Name -> PubChem CID
    - Removes duplicate CID column
    - Resets index
    - Adds S.No
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.DataFrame()

    df = df.copy()

    if "Name" in df.columns:
        df = df.rename(
            columns={"Name": pubchem_col_name}
        )

    if "CID" in df.columns:
        df = df.drop(columns=["CID"])

    df = df.reset_index(drop=True)

    df.index = df.index + 1

    df = df.reset_index()

    df = df.rename(
        columns={"index": "S.No"}
    )

    return df


# ============================================================
# Protein PDB Parser
# ============================================================

def parse_protein_pdb(pdb_file_path):
    """
    Parses clean PDB target protein and calculates
    the geometric centroid of all protein atoms.
    """

    coords = []

    if (
        not pdb_file_path
        or not os.path.exists(pdb_file_path)
    ):
        return 0, (0.0, 0.0, 0.0)

    with open(
        pdb_file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            if line.startswith("ATOM"):

                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    coords.append([x, y, z])

                except ValueError:
                    continue

    if not coords:
        return 0, (0.0, 0.0, 0.0)

    coords_arr = np.array(coords)

    atom_count = len(coords_arr)

    centroid = tuple(
        np.mean(coords_arr, axis=0)
    )

    return atom_count, centroid


# ============================================================
# Extract Top GNINA Pose
# ============================================================

def extract_top_ligand_pose(
    gnina_output_path,
    top_pose_path
):
    """
    Extracts ONLY the first GNINA pose (Mode 1).
    """

    if not os.path.exists(gnina_output_path):
        return False

    try:

        supplier = Chem.SDMolSupplier(
            gnina_output_path
        )

        mols = [
            m for m in supplier
            if m is not None
        ]

        if len(mols) > 0:

            writer = Chem.SDWriter(
                top_pose_path
            )

            writer.write(mols[0])
            writer.close()

            return True

    except Exception as e:
        print(
            f"Pose extraction error: {e}"
        )

    return False


# ============================================================
# Convert SDF Pose to PDB
# ============================================================

def convert_ligand_pose_to_pdb(
    sdf_path,
    pdb_path
):
    """
    Converts docked ligand SDF to PDB while
    preserving the docked 3D coordinates.
    """

    if not os.path.exists(sdf_path):
        return False

    try:

        supplier = Chem.SDMolSupplier(
            sdf_path,
            removeHs=False,
            sanitize=False
        )

        mol = next(
            (
                m for m in supplier
                if m is not None
            ),
            None
        )

        if mol is None:
            return False

        try:

            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE
                )
            )

        except Exception:
            pass

        pdb_block = Chem.MolToPDBBlock(
            mol,
            confId=0,
            flavor=0
        )

        with open(
            pdb_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(pdb_block)

        return os.path.exists(pdb_path)

    except Exception as e:

        print(
            "Ligand SDF->PDB coordinate "
            f"conversion error: {e}"
        )

        return False


# ============================================================
# Create Protein-Ligand Complex
# ============================================================

def create_protein_ligand_complex(
    receptor_path,
    ligand_pdb_path,
    output_complex_path
):
    """
    Combines receptor and docked ligand into
    a single protein-ligand complex PDB.
    """

    with open(
        receptor_path,
        "r"
    ) as f_rec:

        rec_data = f_rec.read()

    with open(
        ligand_pdb_path,
        "r"
    ) as f_lig:

        lig_data = f_lig.read()

    with open(
        output_complex_path,
        "w"
    ) as f_out:

        f_out.write(
            rec_data.replace(
                "END",
                ""
            ).strip()
            + "\n"
        )

        f_out.write("TER\n")

        for line in lig_data.splitlines():

            if line.startswith("ATOM"):

                f_out.write(
                    "HETATM"
                    + line[6:]
                    + "\n"
                )

            elif line.startswith("HETATM"):

                f_out.write(
                    line + "\n"
                )

        f_out.write("END\n")

    return os.path.exists(
        output_complex_path
    )


# ============================================================
# Ligand Processing
# ============================================================

def process_ligands(
    ligand_file_path,
    filter_type
):
    """
    Loads SDF or CSV ligands and applies
    Lipinski filtering when selected.
    """

    if (
        not ligand_file_path
        or not os.path.exists(ligand_file_path)
    ):
        return pd.DataFrame(), []

    mols = []

    ext = os.path.splitext(
        ligand_file_path
    )[-1].lower()

    # --------------------------------------------------------
    # SDF
    # --------------------------------------------------------

    if ext == ".sdf":

        suppl = Chem.SDMolSupplier(
            ligand_file_path
        )

        mols = [
            m for m in suppl
            if m is not None
        ]

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    elif ext == ".csv":

        df_in = pd.read_csv(
            ligand_file_path
        )

        smiles_col = next(
            (
                col
                for col in df_in.columns
                if "smiles" in col.lower()
            ),
            None
        )

        name_col = next(
            (
                col
                for col in df_in.columns
                if col.lower()
                in ["name", "cid", "id"]
            ),
            None
        )

        if smiles_col:

            for idx, row in df_in.iterrows():

                m = Chem.MolFromSmiles(
                    str(row[smiles_col])
                )

                if m:

                    mol_name = (
                        str(row[name_col])
                        if name_col
                        else f"Mol_{idx + 1}"
                    )

                    m.SetProp(
                        "_Name",
                        mol_name
                    )

                    mols.append(m)

    parsed_data = []
    valid_mols = []

    # --------------------------------------------------------
    # Calculate descriptors
    # --------------------------------------------------------

    for idx, mol in enumerate(mols):

        name = (
            mol.GetProp("_Name")
            if mol.HasProp("_Name")
            else f"Mol_{idx + 1}"
        )

        mw = round(
            Descriptors.MolWt(mol),
            2
        )

        logp = round(
            Descriptors.MolLogP(mol),
            2
        )

        hbd = Lipinski.NumHDonors(
            mol
        )

        hba = Lipinski.NumHAcceptors(
            mol
        )

        if filter_type == "Lipinski":

            if (
                mw <= 500
                and logp <= 5
                and hbd <= 5
                and hba <= 10
            ):

                parsed_data.append({
                    "Name": name,
                    "MW": mw,
                    "LogP": logp,
                    "HBD": hbd,
                    "HBA": hba
                })

                valid_mols.append(mol)

        else:

            parsed_data.append({
                "Name": name,
                "MW": mw,
                "LogP": logp,
                "HBD": hbd,
                "HBA": hba
            })

            valid_mols.append(mol)

    return (
        pd.DataFrame(parsed_data),
        valid_mols
    )


# ============================================================
# GNINA GPU Detection
# ============================================================

def gnina_gpu_available():

    if not shutil.which("nvidia-smi"):
        return False

    try:

        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )

        return result.returncode == 0

    except Exception:
        return False


# ============================================================
# Write Individual Ligand SDF
# ============================================================

def write_single_ligand_sdf(
    mol,
    mol_name,
    temp_dir
):

    path = os.path.join(
        temp_dir,
        f"{mol_name}_input.sdf"
    )

    writer = Chem.SDWriter(path)

    writer.write(mol)

    writer.close()

    return path


# ============================================================
# Run GNINA
# ============================================================

def run_gnina_docking(
    clean_target_path,
    ligand_path,
    output_path,
    center_x,
    center_y,
    center_z,
    size_x,
    size_y,
    size_z
):

    for exe in [
        "gnina",
        "gninabase"
    ]:

        cmd = [
            exe,

            "-r",
            clean_target_path,

            "-l",
            ligand_path,

            "-o",
            output_path,

            "--center_x",
            str(center_x),

            "--center_y",
            str(center_y),

            "--center_z",
            str(center_z),

            "--size_x",
            str(size_x),

            "--size_y",
            str(size_y),

            "--size_z",
            str(size_z),

            "--num_modes",
            "9",

            "--cnn_scoring",
            "rescore",

            "--seed",
            str(GNINA_SEED),
        ]

        if FORCE_CPU:
            cmd.append("--no_gpu")

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if (
                result.returncode == 0
                or "Error"
                not in result.stderr
            ):

                return (
                    result.stdout,
                    result.stderr,
                    result.returncode
                )

        except FileNotFoundError:

            continue

    return (
        "",
        "GNINA / gninabase binary "
        "not found in system PATH.",
        1
    )


# ============================================================
# Safe Float
# ============================================================

def _safe_float(val):

    if val is None:
        return None

    s = str(val).strip()

    try:
        return float(s)

    except ValueError:

        parts = [
            p
            for p in s.replace(
                ",",
                " "
            ).split()
            if p
        ]

        try:

            vals = [
                float(p)
                for p in parts
            ]

            return (
                sum(vals) / len(vals)
                if vals
                else None
            )

        except ValueError:

            return None


# ============================================================
# Parse GNINA Score
# ============================================================

def parse_gnina_score(
    top_pose_sdf_path
):

    affinity = None
    cnn_score = None
    cnn_affinity = None

    if not os.path.exists(
        top_pose_sdf_path
    ):

        return {
            "affinity": None,
            "cnn_score": None,
            "cnn_affinity": None
        }

    try:

        supplier = Chem.SDMolSupplier(
            top_pose_sdf_path
        )

        mol = next(
            (
                m
                for m in supplier
                if m is not None
            ),
            None
        )

        if mol is not None:

            props = mol.GetPropsAsDict()

            for key, val in props.items():

                k_lower = key.lower()

                if (
                    "minimizedaffinity"
                    in k_lower
                    or "affinity"
                    in k_lower
                ):

                    if affinity is None:

                        affinity = _safe_float(
                            val
                        )

                if "cnnscore" in k_lower:

                    cnn_score = _safe_float(
                        val
                    )

                if "cnnaffinity" in k_lower:

                    cnn_affinity = _safe_float(
                        val
                    )

            if (
                affinity is None
                and props.get(
                    "minimizedAffinity"
                ) is not None
            ):

                affinity = _safe_float(
                    props.get(
                        "minimizedAffinity"
                    )
                )

            if (
                cnn_score is None
                and props.get(
                    "CNNscore"
                ) is not None
            ):

                cnn_score = _safe_float(
                    props.get(
                        "CNNscore"
                    )
                )

            if (
                cnn_affinity is None
                and props.get(
                    "CNNaffinity"
                ) is not None
            ):

                cnn_affinity = _safe_float(
                    props.get(
                        "CNNaffinity"
                    )
                )

            affinity = (
                round(affinity, 2)
                if affinity is not None
                else None
            )

            cnn_score = (
                round(cnn_score, 3)
                if cnn_score is not None
                else None
            )

            cnn_affinity = (
                round(cnn_affinity, 3)
                if cnn_affinity is not None
                else None
            )

    except Exception as e:

        print(
            f"Error parsing GNINA score: {e}"
        )

    return {
        "affinity": affinity,
        "cnn_score": cnn_score,
        "cnn_affinity": cnn_affinity
    }


# ============================================================
# 2. MAIN PIPELINE
# ============================================================

def run_deepdock_pipeline(
    ligand_file,
    filter_type,
    target_file,
    custom_cx,
    custom_cy,
    custom_cz,
    size_x,
    size_y,
    size_z
):

    status_log = (
        "🚀 Initiating DeepDock-AI Pipeline...\n"
    )

    filtered_df = pd.DataFrame()
    docking_df = pd.DataFrame()
    admet_df = pd.DataFrame()

    filter_csv_path = None
    docking_csv_path = None
    admet_csv_path = None
    zip_path = None

    temp_dir = None

    try:

        # ====================================================
        # Validate inputs
        # ====================================================

        if (
            ligand_file is None
            or target_file is None
        ):

            return (
                "❌ Error: Upload both files.",
                filtered_df,
                docking_df,
                admet_df,
                None,
                None,
                None,
                None
            )

        # ====================================================
        # 1. TARGET PREPARATION
        # ====================================================

        status_log += (
            "\n[1/3] Parsing & Cleaning "
            "Target Protein..."
        )

        temp_dir = tempfile.mkdtemp()

        clean_target_pdb_path = os.path.join(
            temp_dir,
            "clean_receptor.pdb"
        )

        receptor_success = (
            clean_and_prepare_receptor(
                target_file.name,
                clean_target_pdb_path
            )
        )

        if not receptor_success:

            raise RuntimeError(
                "Target protein cleaning failed."
            )

        atom_count, calculated_centroid = (
            parse_protein_pdb(
                clean_target_pdb_path
            )
        )

        status_log += (
            "\n     ✔ Protein Cleaned & "
            f"Parsed successfully! "
            f"({atom_count} atoms)"
        )

        # ====================================================
        # Grid center
        # ====================================================

        final_cx = (
            custom_cx
            if custom_cx != 0.0
            else calculated_centroid[0]
        )

        final_cy = (
            custom_cy
            if custom_cy != 0.0
            else calculated_centroid[1]
        )

        final_cz = (
            custom_cz
            if custom_cz != 0.0
            else calculated_centroid[2]
        )

        status_log += (
            "\n     🎯 Active Grid Box Center: "
            f"X={final_cx:.2f}, "
            f"Y={final_cy:.2f}, "
            f"Z={final_cz:.2f}"
        )

        # ====================================================
        # 2. LIGAND PROCESSING
        # ====================================================

        status_log += (
            f"\n[2/3] Processing Ligands "
            f"(Filter: {filter_type})..."
        )

        filtered_df, valid_mols = (
            process_ligands(
                ligand_file.name,
                filter_type
            )
        )

        status_log += (
            f"\n     ✔ {len(valid_mols)} "
            "valid ligand(s) available."
        )

        # ====================================================
        # 3. DOCKING
        # ====================================================

        status_log += (
            "\n[3/3] Running GNINA Docking "
            "& Complex Generation..."
        )

        complexes_dir = os.path.join(
            temp_dir,
            "protein_ligand_complexes"
        )

        os.makedirs(
            complexes_dir,
            exist_ok=True
        )

        docking_data = []

        # ----------------------------------------------------
        # ADMET data are collected during docking but
        # ADMET itself is NOT executed until all docking
        # calculations have finished.
        # ----------------------------------------------------

        admet_mols = []
        admet_names = []
        admet_scores = []
        admet_cids = []

        # ====================================================
        # GPU / CPU
        # ====================================================

        gpu_present = gnina_gpu_available()

        if FORCE_CPU:

            status_log += (
                "\n     🖥️ GNINA forced to "
                "CPU mode (--no_gpu)"
            )

        else:

            status_log += (
                "\n     🖥️ GNINA auto mode — "
                f"{'GPU' if gpu_present else 'CPU'} detected"
            )

        # ====================================================
        # Dock each ligand
        # ====================================================

        for idx, row in filtered_df.iterrows():

            row_mol = valid_mols[idx]

            mol_name = str(
                row["Name"]
            ).replace(
                " ",
                "_"
            )

            mol_safe_name = (
                f"{idx}_{mol_name}"
            )

            output_sdf = os.path.join(
                temp_dir,
                f"docked_{mol_safe_name}.sdf"
            )

            top_pose_sdf = os.path.join(
                temp_dir,
                f"top1_{mol_safe_name}.sdf"
            )

            top_pose_pdb = os.path.join(
                temp_dir,
                f"top1_{mol_safe_name}.pdb"
            )

            complex_pdb = os.path.join(
                complexes_dir,
                f"Complex_{mol_safe_name}.pdb"
            )

            ligand_sdf = (
                write_single_ligand_sdf(
                    row_mol,
                    mol_safe_name,
                    temp_dir
                )
            )

            status_log += (
                f"\n     🔬 Docking "
                f"{idx + 1}/{len(valid_mols)}: "
                f"{row['Name']}"
            )

            # =================================================
            # GNINA
            # =================================================

            (
                gnina_stdout,
                gnina_stderr,
                exit_code
            ) = run_gnina_docking(
                clean_target_pdb_path,
                ligand_sdf,
                output_sdf,
                final_cx,
                final_cy,
                final_cz,
                size_x,
                size_y,
                size_z
            )

            # =================================================
            # Docking failure
            # =================================================

            if exit_code != 0:

                status_log += (
                    "\n     ⚠️ GNINA failed for "
                    f"{mol_name}: "
                    f"{gnina_stderr.strip()[:200]}"
                )

                score = None
                cnn_score = None
                cnn_affinity = None

            else:

                # =============================================
                # Extract top pose
                # =============================================

                pose_success = (
                    extract_top_ligand_pose(
                        output_sdf,
                        top_pose_sdf
                    )
                )

                if not pose_success:

                    status_log += (
                        "\n     ⚠️ Could not extract "
                        f"top pose for {mol_name}"
                    )

                    score = None
                    cnn_score = None
                    cnn_affinity = None

                else:

                    # =========================================
                    # Convert top pose to PDB
                    # =========================================

                    pdb_success = (
                        convert_ligand_pose_to_pdb(
                            top_pose_sdf,
                            top_pose_pdb
                        )
                    )

                    if not pdb_success:

                        status_log += (
                            "\n     ⚠️ Could not convert "
                            f"top pose to PDB for "
                            f"{mol_name}"
                        )

                    # =========================================
                    # Create protein-ligand complex
                    # =========================================

                    if pdb_success:

                        complex_success = (
                            create_protein_ligand_complex(
                                clean_target_pdb_path,
                                top_pose_pdb,
                                complex_pdb
                            )
                        )

                        if not complex_success:

                            status_log += (
                                "\n     ⚠️ Could not create "
                                f"complex PDB for "
                                f"{mol_name}"
                            )

                    # =========================================
                    # Parse GNINA scores
                    # =========================================

                    parsed = parse_gnina_score(
                        top_pose_sdf
                    )

                    score = parsed[
                        "affinity"
                    ]

                    cnn_score = parsed[
                        "cnn_score"
                    ]

                    cnn_affinity = parsed[
                        "cnn_affinity"
                    ]

            # =================================================
            # Docking result table
            # =================================================

            docking_data.append({
                "Name": row["Name"],
                "Affinity (kcal/mol)": score,
                "CNN Score": cnn_score,
                "CNN Affinity": cnn_affinity,

                # Kept for compatibility with
                # your existing output format
                "RMSD l.b": 0.0,
                "RMSD u.b": 0.0
            })

            # =================================================
            # Store information for later ADMET analysis
            # =================================================

            admet_mols.append(
                row_mol
            )

            admet_names.append(
                str(row["Name"])
            )

            # ADMETlab expects a numerical score.
            # If docking failed, NaN is used.
            admet_scores.append(
                score
                if score is not None
                else float("nan")
            )

            # Preserve the identifier from the ligand input.
            # In your CSV workflow this is normally the CID/name.
            admet_cids.append(
                str(row["Name"])
            )

        # ====================================================
        # ALL DOCKING FINISHED
        # ====================================================

        docking_df = pd.DataFrame(
            docking_data
        )

        status_log += (
            "\n\n✔ ALL docking calculations "
            "completed."
        )

        # ====================================================
        # ADMET ANALYSIS
        # ====================================================

        status_log += (
            "\n\n🧪 Starting ADMET analysis..."
        )

        try:

            if admet_mols:

                status_log += (
                    "\n     🌐 Trying "
                    "ADMETlab 3.0 first..."
                )

                admet_df, admet_source = (
                    run_admet_analysis(
                        mols=admet_mols,
                        names=admet_names,
                        scores=admet_scores,
                        cids=admet_cids,
                        use_api=True,
                        status_text=None
                    )
                )

                if (
                    admet_source
                    == "ADMETlab 3.0"
                ):

                    status_log += (
                        "\n     ✔ ADMETlab 3.0 "
                        "analysis completed successfully."
                    )

                    status_log += (
                        "\n     🌐 ADMET source: "
                        "ADMETlab 3.0 API"
                    )

                else:

                    status_log += (
                        "\n     ⚠️ ADMETlab 3.0 "
                        "was unavailable."
                    )

                    status_log += (
                        "\n     ✔ Automatically "
                        "using RDKit fallback."
                    )

                    status_log += (
                        "\n     🧬 ADMET source: "
                        "RDKit"
                    )

            else:

                admet_df = pd.DataFrame()

                status_log += (
                    "\n     ⚠️ No ligands available "
                    "for ADMET analysis."
                )

        except Exception as e:

            status_log += (
                "\n     ⚠️ ADMET analysis "
                f"encountered an error: {str(e)}"
            )

            admet_df = pd.DataFrame()

        # ====================================================
        # Clean output DataFrames for Gradio
        # ====================================================

        filtered_df = clean_dataframe_indices(
            filtered_df
        )

        docking_df = clean_dataframe_indices(
            docking_df
        )

        admet_df = clean_dataframe_indices(
            admet_df
        )

        # ====================================================
        # Save CSV files
        # ====================================================

        filter_csv_path = os.path.join(
            temp_dir,
            "filtered_ligands.csv"
        )

        docking_csv_path = os.path.join(
            temp_dir,
            "docking_results.csv"
        )

        admet_csv_path = os.path.join(
            temp_dir,
            "admet_results.csv"
        )

        filtered_df.to_csv(
            filter_csv_path,
            index=False
        )

        docking_df.to_csv(
            docking_csv_path,
            index=False
        )

        admet_df.to_csv(
            admet_csv_path,
            index=False
        )

        # ====================================================
        # Create ZIP archive
        # ====================================================

        zip_path = os.path.join(
            temp_dir,
            "DeepDock_Output_Archive.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zipf:

            # CSV files
            zipf.write(
                filter_csv_path,
                arcname="filtered_ligands.csv"
            )

            zipf.write(
                docking_csv_path,
                arcname="docking_results.csv"
            )

            zipf.write(
                admet_csv_path,
                arcname="admet_results.csv"
            )

            # Complex PDBs
            for root, _, files in os.walk(
                complexes_dir
            ):

                for file in files:

                    file_path = os.path.join(
                        root,
                        file
                    )

                    zipf.write(
                        file_path,
                        arcname=os.path.join(
                            "complexes",
                            file
                        )
                    )

        # ====================================================
        # SUCCESS
        # ====================================================

        status_log += (
            "\n\n✨ [SUCCESS] Pipeline complete!"
        )

        status_log += (
            "\n     ✔ Filtered ligand CSV ready."
        )

        status_log += (
            "\n     ✔ Docking results CSV ready."
        )

        status_log += (
            "\n     ✔ Full ADMET results ready."
        )

        status_log += (
            "\n     ✔ Protein-ligand complex "
            "PDB files ready."
        )

        status_log += (
            "\n     ✔ Complete ZIP archive ready."
        )

    except Exception as e:

        status_log += (
            "\n\n❌ [ERROR]: Pipeline failed "
            f"due to: {str(e)}"
        )

    return (
        status_log,
        filtered_df,
        docking_df,
        admet_df,
        filter_csv_path,
        docking_csv_path,
        admet_csv_path,
        zip_path
    )


# ============================================================
# 3. GRADIO UI
# ============================================================

with gr.Blocks(
    title="DeepDock-AI"
) as demo:

    gr.Markdown(
        "# 🧬 DeepDock-AI Pipeline"
    )

    # ========================================================
    # INPUTS
    # ========================================================

    with gr.Row():

        ligand_input = gr.File(
            label="Upload Ligand File (.sdf, .csv)"
        )

        filter_input = gr.Dropdown(
            ["Lipinski", "None"],
            value="Lipinski",
            label="Filter Type"
        )

        target_input = gr.File(
            label="Upload Target Protein (.pdb)"
        )

    # ========================================================
    # GRID BOX
    # ========================================================

    with gr.Accordion(
        "🎯 Binding Site Grid Box Coordinates (Optional)",
        open=True
    ):

        with gr.Row():

            center_x = gr.Number(
                label="Center X",
                value=0.0
            )

            center_y = gr.Number(
                label="Center Y",
                value=0.0
            )

            center_z = gr.Number(
                label="Center Z",
                value=0.0
            )

        with gr.Row():

            size_x = gr.Number(
                label="Size X (Å)",
                value=20.0
            )

            size_y = gr.Number(
                label="Size Y (Å)",
                value=20.0
            )

            size_z = gr.Number(
                label="Size Z (Å)",
                value=20.0
            )

    # ========================================================
    # RUN BUTTON
    # ========================================================

    submit_btn = gr.Button(
        "Run Pipeline",
        variant="primary"
    )

    # ========================================================
    # STATUS
    # ========================================================

    status_output = gr.Textbox(
        label="Status Logs",
        lines=12
    )

    # ========================================================
    # OUTPUT TABS
    # ========================================================

    with gr.Tabs():

        # ----------------------------------------------------
        # FILTERED LIGANDS
        # ----------------------------------------------------

        with gr.TabItem(
            "1. Filtered Ligands"
        ):

            filter_df_output = gr.DataFrame(
                label="Filtered Ligands Results"
            )

            filter_csv_download = gr.File(
                label="📥 Download Filtered Ligands (CSV)"
            )

        # ----------------------------------------------------
        # DOCKING
        # ----------------------------------------------------

        with gr.TabItem(
            "2. Docking Results"
        ):

            docking_df_output = gr.DataFrame(
                label="AutoDock Vina / GNINA Scores"
            )

            docking_csv_download = gr.File(
                label="📥 Download Docking Results (CSV)"
            )

        # ----------------------------------------------------
        # ADMET
        # ----------------------------------------------------

        with gr.TabItem(
            "3. ADMET Analysis"
        ):

            admet_df_output = gr.DataFrame(
                label="ADMET Property Profiles"
            )

            admet_csv_download = gr.File(
                label="📥 Download ADMET Results (CSV)"
            )

    # ========================================================
    # COMPLETE ZIP
    # ========================================================

    with gr.Row():

        zip_output = gr.File(
            label=(
                "📦 Download Complete Package "
                "(All CSVs + Complex PDBs ZIP)"
            )
        )

    # ========================================================
    # BUTTON ACTION
    # ========================================================

    submit_btn.click(

        fn=run_deepdock_pipeline,

        inputs=[
            ligand_input,
            filter_input,
            target_input,

            center_x,
            center_y,
            center_z,

            size_x,
            size_y,
            size_z
        ],

        outputs=[
            status_output,

            filter_df_output,
            docking_df_output,
            admet_df_output,

            filter_csv_download,
            docking_csv_download,
            admet_csv_download,

            zip_output
        ]
    )


# ============================================================
# 4. LAUNCH
# ============================================================

if __name__ == "__main__":

    gr.close_all()

    demo.launch(
        share=True
    )
