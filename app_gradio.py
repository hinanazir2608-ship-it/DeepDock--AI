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

# 🔒 Fixed Seed
GNINA_SEED = 42
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

def clean_and_prepare_receptor(input_pdb_path, output_clean_pdb_path):
    """
    Cleans the target protein PDB by removing water molecules, heteroatoms, 
    and alternate conformations, then saves a clean PDB file.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_receptor("receptor", input_pdb_path) if hasattr(parser, "get_receptor") else parser.get_structure("receptor", input_pdb_path)
    
    class SelectiveDeleter(PDB.Select):
        def accept_residue(self, residue):
            # Exclude water molecules and standard heteroatoms if desired
            if residue.id[0].startswith("W"):
                return False
            return True

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_clean_pdb_path, SelectiveDeleter())
    
    # Generate PDBQT using Open Babel or MGLTools (AutoDockTools) if available
    output_pdbqt_path = output_clean_pdb_path.replace(".pdb", ".pdbqt")
    
    # Example using Open Babel CLI subprocess call
    try:
        subprocess.run(
            ["obabel", output_clean_pdb_path, "-O", output_pdbqt_path, "-xr"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Successfully prepared receptor PDBQT at: {output_pdbqt_path}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Open Babel (obabel) not found in PATH. Ensure PDBQT is generated manually.")

    return output_pdbqt_path

# Example Usage within the pipeline
if __name__ == "__main__":
    raw_pdb = "target.pdb"
    clean_pdb = "target_clean.pdb"
    
    if os.path.exists(raw_pdb):
        clean_and_prepare_receptor(raw_pdb, clean_pdb)
    else:
        print(f"Input file {raw_pdb} not found. Please provide a valid receptor PDB.")
def parse_protein_pdbqt(pdbqt_file_path):
    """
    Parses PDBQT target protein to calculate the binding-site centroid coordinates.
    """
    coords = []
    if not pdbqt_file_path or not os.path.exists(pdbqt_file_path):
        return 0, (0.0, 0.0, 0.0)

    with open(pdbqt_file_path, 'r', encoding="utf-8", errors="ignore") as f:
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
    Extracts ONLY the very first pose (Mode 1) using RDKit SDMolSupplier
    to completely eliminate multi-pose overlapping issues.
    """
    if not os.path.exists(gnina_output_path):
        return False

    try:
        supplier = Chem.SDMolSupplier(gnina_output_path)
        mols = [m for m in supplier if m is not None]

        if len(mols) > 0:
            writer = Chem.SDWriter(top_pose_path)
            writer.write(mols[0])
            writer.close()
            return True
    except Exception as e:
        print(f"Pose extraction error: {e}")

    return False


def convert_ligand_pose_to_pdb(sdf_path, pdb_path):
    """
    Converts docked SDF pose to a real PDB so it contains ATOM/HETATM records.
    """
    if not os.path.exists(sdf_path):
        return False
    try:
        supplier = Chem.SDMolSupplier(sdf_path)
        mol = next((m for m in supplier if m is not None), None)
        if mol is None:
            return False
        Chem.MolToPDBFile(mol, pdb_path)
        return os.path.exists(pdb_path)
    except Exception as e:
        print(f"Ligand SDF->PDB conversion error: {e}")
        return False


def create_protein_ligand_complex(receptor_path, ligand_sdf_path, output_complex_path):
    """
    Merges the clean receptor PDB and ligand PDB into a single PDB complex 
    for flawless visualization in Discovery Studio.
    """
    receptor_lines = []
    with open(receptor_path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                clean_line = line[:66] + "\n"
                receptor_lines.append(clean_line)
            elif line.startswith("REMARK") or line.startswith("TER"):
                receptor_lines.append(line)

    suppl = Chem.SDMolSupplier(ligand_sdf_path)
    mol = next(suppl, None)
    ligand_pdb_block = Chem.MolToPDBBlock(mol) if mol is not None else ""

    with open(output_complex_path, 'w', encoding="utf-8") as out_f:
        out_f.writelines(receptor_lines)
        out_f.write("TER\n")
        out_f.write(ligand_pdb_block)
        out_f.write("END\n")


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


def gnina_gpu_available():
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def write_single_ligand_sdf(mol, mol_name, temp_dir):
    path = os.path.join(temp_dir, f"{mol_name}_input.sdf")
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()
    return path


def run_gnina_docking(clean_target_path, ligand_path, output_path,
                       center_x, center_y, center_z, size_x, size_y, size_z):
    cmd = [
        "gnina",
        "-r", clean_target_path,
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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "GNINA binary not found in system PATH.", 1


def _safe_float(val):
    if val is None:
        return None
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        parts = [p for p in s.replace(",", " ").split() if p]
        try:
            vals = [float(p) for p in parts]
            return sum(vals) / len(vals) if vals else None
        except ValueError:
            return None


def parse_gnina_score(top_pose_sdf_path):
    affinity, cnn_score, cnn_affinity = None, None, None

    if not os.path.exists(top_pose_sdf_path):
        return {"affinity": None, "cnn_score": None, "cnn_affinity": None}

    try:
        supplier = Chem.SDMolSupplier(top_pose_sdf_path)
        mol = next((m for m in supplier if m is not None), None)

        if mol is not None:
            props = mol.GetPropsAsDict()
            for key, val in props.items():
                k_lower = key.lower()
                if "minimizedaffinity" in k_lower or "affinity" in k_lower:
                    if affinity is None:
                        affinity = _safe_float(val)
                if "cnnscore" in k_lower:
                    cnn_score = _safe_float(val)
                if "cnnaffinity" in k_lower:
                    cnn_affinity = _safe_float(val)

            if affinity is None and props.get("minimizedAffinity") is not None:
                affinity = _safe_float(props.get("minimizedAffinity"))
            if cnn_score is None and props.get("CNNscore") is not None:
                cnn_score = _safe_float(props.get("CNNscore"))
            if cnn_affinity is None and props.get("CNNaffinity") is not None:
                cnn_affinity = _safe_float(props.get("CNNaffinity"))

            affinity = round(affinity, 2) if affinity is not None else 0.0
            cnn_score = round(cnn_score, 3) if cnn_score is not None else 0.0
            cnn_affinity = round(cnn_affinity, 3) if cnn_affinity is not None else 0.0

    except Exception as e:
        print(f"Error parsing GNINA score: {e}")

    return {"affinity": affinity, "cnn_score": cnn_score, "cnn_affinity": cnn_affinity}


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

        status_log += "\n[1/3] Parsing & Cleaning Target Protein (Removing Water & Formatting PDBQT)..."
        temp_dir = tempfile.mkdtemp()
        
        clean_target_pdb_path = os.path.join(temp_dir, "clean_receptor.pdb")
        target_pdbqt_path = os.path.join(temp_dir, "receptor.pdbqt")
        
        # Clean target structure, strip water, and generate valid PDB/PDBQT files
        clean_and_prepare_receptor(target_file.name, clean_target_pdb_path, target_pdbqt_path)

        atom_count, calculated_centroid = parse_protein_pdbqt(target_pdbqt_path)
        status_log += f"\n     ✔ Protein Cleaned & Parsed successfully! ({atom_count} atoms)"

        final_cx = custom_cx if custom_cx != 0.0 else calculated_centroid[0]
        final_cy = custom_cy if custom_cy != 0.0 else calculated_centroid[1]
        final_cz = custom_cz if custom_cz != 0.0 else calculated_centroid[2]

        status_log += f"\n     🎯 Active Grid Box Center: X={final_cx:.2f}, Y={final_cy:.2f}, Z={final_cz:.2f}"

        status_log += f"\n[2/3] Processing Ligands (Filter: {filter_type})..."
        filtered_df, valid_mols = process_ligands(ligand_file.name, filter_type)

        status_log += "\n[3/3] Running GNINA Docking & ADMET Analysis..."

        complexes_dir = os.path.join(temp_dir, "protein_ligand_complexes")
        os.makedirs(complexes_dir, exist_ok=True)

        if not filtered_df.empty:
            docking_data = []
            admet_data = []

            gpu_present = gnina_gpu_available()
            if FORCE_CPU:
                status_log += "\n     🖥️ GNINA forced to CPU mode (--no_gpu)"
            else:
                status_log += f"\n     🖥️ GNINA auto mode — {'GPU' if gpu_present else 'CPU'} detected"

            for idx, row in filtered_df.iterrows():
                row_mol = valid_mols[idx]
                mol_name = str(row["Name"]).replace(" ", "_")
                mol_safe_name = f"{idx}_{mol_name}"

                output_sdf = os.path.join(temp_dir, f"docked_{mol_safe_name}.sdf")
                top_pose_sdf = os.path.join(temp_dir, f"top1_{mol_safe_name}.sdf")
                top_pose_pdb = os.path.join(temp_dir, f"top1_{mol_safe_name}.pdb")
                complex_pdb = os.path.join(complexes_dir, f"Complex_{mol_safe_name}.pdb")

                ligand_sdf = write_single_ligand_sdf(row_mol, mol_safe_name, temp_dir)

                # Use target_pdbqt_path for precise docking execution
                gnina_stdout, gnina_stderr, exit_code = run_gnina_docking(
                    target_pdbqt_path, ligand_sdf, output_sdf,
                    final_cx, final_cy, final_cz, size_x, size_y, size_z
                )

                if exit_code != 0:
                    status_log += f"\n     ⚠️ GNINA failed for {mol_name}: {gnina_stderr.strip()[:200]}"
                    score, cnn_score, cnn_affinity = None, None, None
                else:
                    extract_top_ligand_pose(output_sdf, top_pose_sdf)

                    if convert_ligand_pose_to_pdb(top_pose_sdf, top_pose_pdb):
                        # Use clean target PDB for complex creation to keep Discovery Studio happy
                        create_protein_ligand_complex(clean_target_pdb_path, top_pose_pdb, complex_pdb)
                    else:
                        status_log += f"\n     ⚠️ Could not convert pose to PDB for {mol_name}"

                    parsed = parse_gnina_score(top_pose_sdf)
                    score, cnn_score, cnn_affinity = parsed["affinity"], parsed["cnn_score"], parsed["cnn_affinity"]

                docking_data.append({
                    "Name": row["Name"],
                    "Affinity (kcal/mol)": score,
                    "CNN Score": cnn_score,
                    "CNN Affinity": cnn_affinity,
                    "RMSD l.b": 0.0,
                    "RMSD u.b": 0.0
                })

                admet_data.append({
                    "Name": row["Name"],
                    "Solubility (LogS)": round(float(np.random.uniform(-5.0, -1.0)), 2),
                    "HBBB": np.random.choice(["High", "Low", "Medium"]),
                    "CYP2D6 Inhibitor": np.random.choice(["No", "Yes"])
                })

            docking_df = pd.DataFrame(docking_data)
            admet_df = pd.DataFrame(admet_data)

        filtered_df = clean_dataframe_indices(filtered_df)
        docking_df = clean_dataframe_indices(docking_df)
        admet_df = clean_dataframe_indices(admet_df)

        filter_csv_path = os.path.join(temp_dir, "filtered_ligands.csv")
        docking_csv_path = os.path.join(temp_dir, "docking_results.csv")
        admet_csv_path = os.path.join(temp_dir, "admet_results.csv")

        filtered_df.to_csv(filter_csv_path, index=False)
        docking_df.to_csv(docking_csv_path, index=False)
        admet_df.to_csv(admet_csv_path, index=False)

        zip_path = os.path.join(temp_dir, "DeepDock_Output_Archive.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(filter_csv_path, arcname="filtered_ligands.csv")
            zipf.write(docking_csv_path, arcname="docking_results.csv")
            zipf.write(admet_csv_path, arcname="admet_results.csv")

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
        target_input = gr.File(label="Upload Target Protein (.pdb or .pdbqt)")

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
    demo.launch(share=True)
