import os
import shutil
import zipfile
import tempfile
import subprocess
import numpy as np
import pandas as pd
import gradio as gr

# RDKit Logging & Import Fixes
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

GNINA_SEED = 42
# True  = always pass --no_gpu, so GNINA runs identical CPU-only math on Kaggle
#         AND on your local Ubuntu box, regardless of which one has a GPU.
#         Slower, but removes GPU-vs-CPU as a source of cross-platform difference.
# False = let GNINA auto-use a GPU when it finds one (faster on Kaggle), but then
#         Kaggle and a GPU-less Ubuntu box are no longer running the identical
#         code path for CNN scoring.
FORCE_CPU = False


# ==========================================
# 1. HELPER & PREPROCESSING FUNCTIONS
# ==========================================

def clean_dataframe_indices(df, pubchem_col_name="PubChem CID"):
    """
    Cleans DataFrames for Gradio UI display:
    - Renames 'Name' column to 'PubChem CID'.
    - Drops any existing 'CID' column to prevent duplication.
    - Resets index cleanly to start from 1, 2, 3...
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "Name" in df.columns:
        df = df.rename(columns={"Name": pubchem_col_name})

    if "CID" in df.columns:
        df = df.drop(columns=["CID"])

    df = df.reset_index(drop=True)
    df.index = df.index + 1

    df = df.reset_index()
    df = df.rename(columns={"index": "S.No"})

    return df


def parse_protein_pdbqt(pdbqt_file_path):
    """
    Parses PDBQT target protein to calculate the binding-site centroid coordinates.
    """
    coords = []
    if not pdbqt_file_path or not os.path.exists(pdbqt_file_path):
        return 0, (0.0, 0.0, 0.0)

    with open(pdbqt_file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
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
    centroid = tuple(np.mean(coords_arr, axis=0))

    return atom_count, centroid


def extract_top_ligand_pose(gnina_output_path, top_pose_path):
    """
    Extracts ONLY Mode 1 (RMSD l.b = 0.0, RMSD u.b = 0.0) from GNINA multi-pose output.
    """
    if not os.path.exists(gnina_output_path):
        return False

    ext = os.path.splitext(gnina_output_path)[-1].lower()

    with open(gnina_output_path, 'r') as infile, open(top_pose_path, 'w') as outfile:
        if ext == ".sdf":
            for line in infile:
                outfile.write(line)
                if line.strip() == "$$$$":  # First SDF entry ends here
                    break
        else:
            in_model = False
            for line in infile:
                if line.startswith("MODEL 1") or line.startswith("MODEL    1"):
                    in_model = True
                elif line.startswith("ENDMDL"):
                    outfile.write("ENDMDL\n")
                    break

                if in_model or not line.startswith("MODEL"):
                    outfile.write(line)
                    if line.startswith("ENDMDL"):
                        break
    return True


def create_protein_ligand_complex(receptor_pdbqt, top_ligand_path, output_complex_pdb):
    """
    Merges protein receptor and top ligand pose while explicitly removing
    any residual water molecules (HOH, WAT, SOL, TIP3).
    """
    water_res_names = {"HOH", "WAT", "SOL", "TIP3"}

    try:
        with open(output_complex_pdb, 'w') as complex_file:
            if os.path.exists(receptor_pdbqt):
                with open(receptor_pdbqt, 'r') as f_rec:
                    for line in f_rec:
                        if line.startswith(("ATOM", "HETATM")):
                            res_name = line[17:20].strip()
                            if res_name not in water_res_names:
                                complex_file.write(line)

            complex_file.write("TER\n")

            if os.path.exists(top_ligand_path):
                with open(top_ligand_path, 'r') as f_lig:
                    for line in f_lig:
                        if line.startswith(("ATOM", "HETATM")):
                            res_name = line[17:20].strip()
                            if res_name not in water_res_names:
                                complex_file.write(line)

            complex_file.write("END\n")
        return True
    except Exception as e:
        print(f"Complex generation error: {e}")
        return False


def process_ligands(ligand_file_path, filter_type):
    """
    RDKit-based ligand loading and Lipinski Rule of 5 filtering.
    """
    if not ligand_file_path or not os.path.exists(ligand_file_path):
        return pd.DataFrame(), []

    mols = []
    ext = os.path.splitext(ligand_file_path)[-1].lower()

    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(ligand_file_path)
        mols = [m for m in suppl if m is not None]
    elif ext == ".csv":
        df_in = pd.read_csv(ligand_file_path)
        smiles_col = next((col for col in df_in.columns if "smiles" in col.lower()), None)
        name_col = next((col for col in df_in.columns if col.lower() in ["name", "cid", "id"]), None)

        if smiles_col:
            for idx, row in df_in.iterrows():
                m = Chem.MolFromSmiles(str(row[smiles_col]))
                if m:
                    mol_name = str(row[name_col]) if name_col else f"Mol_{idx+1}"
                    m.SetProp("_Name", mol_name)
                    mols.append(m)

    parsed_data = []
    valid_mols = []

    for idx, mol in enumerate(mols):
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol_{idx+1}"
        mw = round(Descriptors.MolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 2)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        if filter_type == "Lipinski":
            if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                parsed_data.append({"Name": name, "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba})
                valid_mols.append(mol)
        else:
            parsed_data.append({"Name": name, "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba})
            valid_mols.append(mol)

    return pd.DataFrame(parsed_data), valid_mols


# ---- GNINA scoring fix starts here ----

def gnina_gpu_available():
    """
    Best-effort GPU presence check — used only for the status-log message
    below, not to decide the command line (see FORCE_CPU).
    """
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def write_single_ligand_sdf(mol, mol_name, temp_dir):
    """Writes ONE molecule to its own SDF so GNINA docks exactly this ligand
    (previously the whole uploaded ligand_file was re-docked on every loop
    iteration regardless of which row was being processed)."""
    path = os.path.join(temp_dir, f"{mol_name}_input.sdf")
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()
    return path


def run_gnina_docking(receptor_path, ligand_path, output_path,
                       center_x, center_y, center_z, size_x, size_y, size_z):
    cmd = [
        "gnina",
        "-r", receptor_path,
        "-l", ligand_path,
        "-o", output_path,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--num_modes", "9",
        "--cnn_scoring", "rescore",
        "--seed", str(GNINA_SEED),
    ]
    if FORCE_CPU:
        cmd.append("--no_gpu")
    # else: no flag needed — GNINA uses a GPU automatically if it detects one.

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "GNINA binary not found in system PATH.", 1


def parse_gnina_score(top_pose_sdf_path):
    """
    Reads the REAL score GNINA wrote into the docked pose's SDF properties
    instead of fabricating one. Property names can shift slightly across
    GNINA builds — print mol.GetPropsAsDict() once to confirm on your version
    if this comes back empty.
    """
    affinity, cnn_score = None, None

    if not os.path.exists(top_pose_sdf_path):
        return {"affinity": None, "cnn_score": None}

    supplier = Chem.SDMolSupplier(top_pose_sdf_path)
    mol = next((m for m in supplier if m is not None), None)

    if mol is not None:
        props = mol.GetPropsAsDict()
        if "minimizedAffinity" in props:
            affinity = round(float(props["minimizedAffinity"]), 2)
        if "CNNscore" in props:
            cnn_score = round(float(props["CNNscore"]), 3)

    return {"affinity": affinity, "cnn_score": cnn_score}

# ---- GNINA scoring fix ends here ----


# ==========================================
# 2. MAIN PIPELINE EXECUTION FUNCTION
# ==========================================

def run_deepdock_pipeline(ligand_file, filter_type, target_file, custom_cx, custom_cy, custom_cz, size_x, size_y, size_z):
    status_log = "🚀 Initiating DeepDock-AI Pipeline...\n"

    filtered_df = pd.DataFrame()
    docking_df = pd.DataFrame()
    admet_df = pd.DataFrame()

    filter_csv_path = None
    docking_csv_path = None
    admet_csv_path = None
    zip_path = None

    try:
        if ligand_file is None or target_file is None:
            return "❌ Error: Upload both files.", filtered_df, docking_df, admet_df, None, None, None, None

        # Step 1: Target Protein Parsing
        status_log += "\n[1/3] Parsing Target Protein..."
        atom_count, calculated_centroid = parse_protein_pdbqt(target_file.name)
        status_log += f"\n     ✔ Protein Parsed successfully!"

        final_cx = custom_cx if custom_cx != 0.0 else calculated_centroid[0]
        final_cy = custom_cy if custom_cy != 0.0 else calculated_centroid[1]
        final_cz = custom_cz if custom_cz != 0.0 else calculated_centroid[2]

        status_log += f"\n     🎯 Active Grid Box Center: X={final_cx:.2f}, Y={final_cy:.2f}, Z={final_cz:.2f}"

        # Step 2: Ligand Filtering
        status_log += f"\n[2/3] Processing Ligands (Filter: {filter_type})..."
        filtered_df, valid_mols = process_ligands(ligand_file.name, filter_type)

        # Step 3: Docking, Pose Extraction & Complex Generation
        status_log += "\n[3/3] Running GNINA Docking & ADMET Analysis..."

        temp_dir = tempfile.mkdtemp()
        complexes_dir = os.path.join(temp_dir, "protein_ligand_complexes")
        os.makedirs(complexes_dir, exist_ok=True)

        if not filtered_df.empty:
            docking_data = []
            admet_data = []

            # Logged once, not per-ligand, so GPU/CPU mode is visible instead
            # of silently differing between machines.
            gpu_present = gnina_gpu_available()
            if FORCE_CPU:
                status_log += "\n     🖥️ GNINA forced to CPU mode (--no_gpu) for cross-platform consistency"
                if gpu_present:
                    status_log += " — GPU was available but intentionally not used"
            else:
                status_log += f"\n     🖥️ GNINA auto mode — {'GPU' if gpu_present else 'CPU'} detected"

            for idx, row in filtered_df.iterrows():
               # Loop ke andar yeh line update karein:
                mol_safe_name = f"{idx+1}_{str(row['Name']).replace(' ', '_')}"

                output_sdf = os.path.join(temp_dir, f"{mol_safe_name}_docked.sdf")
                top_pose_sdf = os.path.join(temp_dir, f"{mol_safe_name}_top1.sdf")
                complex_pdb = os.path.join(complexes_dir, f"Complex_{mol_safe_name}_Mode1.pdb")

                ligand_sdf = write_single_ligand_sdf(row_mol, mol_safe_name, temp_dir)

                # Dock ONLY this molecule (was ligand_file.name — the whole
                # uploaded file — re-docked on every iteration before)
                ligand_sdf = write_single_ligand_sdf(row_mol, mol_name, temp_dir)

                gnina_stdout, gnina_stderr, exit_code = run_gnina_docking(
                    target_file.name, ligand_sdf, output_sdf,
                    final_cx, final_cy, final_cz, size_x, size_y, size_z
                )

                if exit_code != 0:
                    status_log += f"\n     ⚠️ GNINA failed for {mol_name}: {gnina_stderr.strip()[:200]}"
                    score, cnn_score = None, None
                else:
                    extract_top_ligand_pose(output_sdf, top_pose_sdf)
                    create_protein_ligand_complex(target_file.name, top_pose_sdf, complex_pdb)
                    parsed = parse_gnina_score(top_pose_sdf)
                    score, cnn_score = parsed["affinity"], parsed["cnn_score"]

                docking_data.append({
                    "Name": mol_name,
                    "Affinity (kcal/mol)": score,
                    "CNN Score": cnn_score,
                    "RMSD l.b": 0.0,
                    "RMSD u.b": 0.0
                })

                # NOTE: ADMET values below are still np.random placeholders —
                # not fixed here, needs a real ADMET method wired in separately.
                admet_data.append({
                    "Name": mol_name,
                    "Solubility (LogS)": round(float(np.random.uniform(-5.0, -1.0)), 2),
                    "HBBB": np.random.choice(["High", "Low", "Medium"]),
                    "CYP2D6 Inhibitor": np.random.choice(["No", "Yes"])
                })

            docking_df = pd.DataFrame(docking_data)
            admet_df = pd.DataFrame(admet_data)

        # Index Cleanups
        filtered_df = clean_dataframe_indices(filtered_df)
        docking_df = clean_dataframe_indices(docking_df)
        admet_df = clean_dataframe_indices(admet_df)

        # Generate CSV files for individual tables
        filter_csv_path = os.path.join(temp_dir, "filtered_ligands.csv")
        docking_csv_path = os.path.join(temp_dir, "docking_results.csv")
        admet_csv_path = os.path.join(temp_dir, "admet_results.csv")

        filtered_df.to_csv(filter_csv_path, index=False)
        docking_df.to_csv(docking_csv_path, index=False)
        admet_df.to_csv(admet_csv_path, index=False)

        # Create combined ZIP Archive (CSVs + Complex PDBs)
        zip_path = os.path.join(temp_dir, "DeepDock_Output_Archive.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(filter_csv_path, arcname="filtered_ligands.csv")
            zipf.write(docking_csv_path, arcname="docking_results.csv")
            zipf.write(admet_csv_path, arcname="admet_results.csv")

            # Pack all generated protein-ligand complexes
            for root, _, files in os.walk(complexes_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=os.path.join("complexes", file))

        status_log += "\n\n✨ [SUCCESS] Pipeline complete! All CSV files & ZIP ready for download."

    except Exception as e:
        status_log += f"\n\n❌ [ERROR]: Pipeline failed due to: {str(e)}"

    return status_log, filtered_df, docking_df, admet_df, filter_csv_path, docking_csv_path, admet_csv_path, zip_path


# ==========================================
# 3. GRADIO UI INTERFACE
# ==========================================

with gr.Blocks(title="DeepDock-AI") as demo:
    gr.Markdown("# 🧬 DeepDock-AI Pipeline")

    with gr.Row():
        ligand_input = gr.File(label="Upload Ligand File (.sdf, .csv)")
        filter_input = gr.Dropdown(["Lipinski", "None"], value="Lipinski", label="Filter Type")
        target_input = gr.File(label="Upload Target PDBQT")

    with gr.Accordion("🎯 Binding Site Grid Box Coordinates (Optional)", open=True):
        with gr.Row():
            center_x = gr.Number(label="Center X", value=0.0)
            center_y = gr.Number(label="Center Y", value=0.0)
            center_z = gr.Number(label="Center Z", value=0.0)
        with gr.Row():
            size_x = gr.Number(label="Size X (Å)", value=20.0)
            size_y = gr.Number(label="Size Y (Å)", value=20.0)
            size_z = gr.Number(label="Size Z (Å)", value=20.0)

    submit_btn = gr.Button("Run Pipeline", variant="primary")

    status_output = gr.Textbox(label="Status Logs", lines=7)

    # --- TABS WITH SPECIFIC DOWNLOAD BUTTONS ---
    with gr.Tabs():
        with gr.TabItem("1. Filtered Ligands"):
            filter_df_output = gr.DataFrame(label="Filtered Ligands Results")
            filter_csv_download = gr.File(label="📥 Download Filtered Ligands (CSV)")

        with gr.TabItem("2. Docking Results"):
            docking_df_output = gr.DataFrame(label="AutoDock Vina / GNINA Scores (Best Mode RMSD=0)")
            docking_csv_download = gr.File(label="📥 Download Docking Results (CSV)")

        with gr.TabItem("3. ADMET Analysis"):
            admet_df_output = gr.DataFrame(label="ADMET Property Profiles")
            admet_csv_download = gr.File(label="📥 Download ADMET Results (CSV)")

    # --- MAIN ZIP DOWNLOAD ---
    with gr.Row():
        zip_output = gr.File(label="📦 Download Complete Package (All CSVs + Complex PDBs ZIP)")

    submit_btn.click(
        fn=run_deepdock_pipeline,
        inputs=[
            ligand_input, filter_input, target_input,
            center_x, center_y, center_z,
            size_x, size_y, size_z
        ],
        outputs=[
            status_output,
            filter_df_output, docking_df_output, admet_df_output,
            filter_csv_download, docking_csv_download, admet_csv_download,
            zip_output
        ]
    )

if __name__ == "__main__":
    gr.close_all()
    demo = create_app()
    demo.launch(share=True)
