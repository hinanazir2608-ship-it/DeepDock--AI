import os

gradio_code ='''import os
import sys
import zipfile
import pandas as pd
import gradio as gr

# Ensure modules directory path is in Python path
modules_path = os.path.join(os.getcwd(), "modules")
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Import DeepDock-AI Pipeline Modules
from filters import apply_rdkit_filters
from docking import run_batched_docking
from admetlab import run_admet_analysis
from export import create_results_zip

def run_deepdock_pipeline(ligand_file, protein_file, batch_size, filter_type, use_admet_api, progress=gr.Progress()):
    if ligand_file is None or protein_file is None:
        return "❌ Error: Please upload both Ligand (.sdf) and Protein (.pdb) files.", None, None, None, None

    status_log = "🚀 Starting DeepDock-AI Full Virtual Screening Pipeline...\\n"
    progress(0.1, desc="Processing Input Files...")
    
    import os
from rdkit import Chem
import pandas as pd


def apply_rdkit_filters(file_path, filter_type="Lipinski"):
    mols = []

    # 1. Check file extension
    ext = os.path.splitext(file_path)[-1].lower()

    # 2. SDF / MOL file handling
    if ext in [".sdf", ".mol"]:
        supplier = Chem.SDMolSupplier(file_path)
        mols = [m for m in supplier if m is not None]

    # 3. CSV / TXT file handling (Jahan SMILES column ho)
    elif ext in [".csv", ".txt", ".smi"]:
        df = pd.read_csv(file_path)

        # SMILES wala column dhoondain
        smiles_col = None
        for col in df.columns:
            if "smiles" in col.lower() or "smi" in col.lower():
                smiles_col = col
                break

        if smiles_col:
            mols = [
                Chem.MolFromSmiles(str(s))
                for s in df[smiles_col]
                if pd.notna(s) and Chem.MolFromSmiles(str(s)) is not None
            ]
        else:
            # Agar single SMILES per line file ho
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip().split()[0]  # First column as SMILES
                    m = Chem.MolFromSmiles(line)
                    if m:
                        mols.append(m)

    # -------------------------------------------------------------
    # Filtering logic here...
    # -------------------------------------------------------------
    # ✅ Safe method: Agar name nahi hai to auto-generate kar dein
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
         mol_name = mol.GetProp("_Name").strip()
        else:
         mol_name = f"Molecule_{idx + 1}"  # Automatic default name (e.g., Molecule_1, Molecule_2)

   # Ab mol object par descriptor filters apply karein:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
# ... baaki filtering code
    filtered_mols = []
    filtered_names = []
    filtered_cids = []

    # Example filter loop
    for idx, mol in enumerate(mols):
        # Apply your Lipinski/Veber/etc filters on 'mol' object
        filtered_mols.append(mol)
        filtered_names.append(
            mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol_{idx+1}"
        )
        filtered_cids.append(idx + 1)

    filter_df = pd.DataFrame({"Name": filtered_names, "CID": filtered_cids})

    return filtered_mols, filtered_names, filtered_cids, filter_df

    # -------------------------------------------------------------
    # Stage 2: Batched Molecular Docking & GNINA/Vina Scoring
    # -------------------------------------------------------------
    status_log += f"⚡ Stage 2: Executing Batched Molecular Docking (Batch Size: {batch_size})...\\n"
    progress(0.55, desc="Stage 2: Batched Molecular Docking")
    
    try:
        docking_results, docked_mols, docking_scores = run_batched_docking(
            filtered_mols, filtered_names, protein_file.name, batch_size=int(batch_size)
        )
        status_log += f"✅ Stage 2 Complete: Successfully docked {len(docked_mols)} complexes.\\n"
    except Exception as e:
        return f"❌ Stage 2 Error: {str(e)}", filter_df, None, None, None

    # -------------------------------------------------------------
    # Stage 3: ADMETlab 3.0 Profiling & ADMET Descriptors
    # -------------------------------------------------------------
    status_log += "🧬 Stage 3: Running ADMETlab 3.0 & ADMET Pharmacokinetics Profiling...\\n"
    progress(0.80, desc="Stage 3: ADMETlab Profiling")
    
    try:
        admet_df, admet_source = run_admet_analysis(
            docked_mols, filtered_names, docking_scores, cids=filtered_cids, use_api=use_admet_api
        )
        status_log += f"✅ Stage 3 Complete: ADMET calculated via ({admet_source}).\\n"
    except Exception as e:
        admet_df = pd.DataFrame({"Error": [str(e)]})
        status_log += f"⚠️ Stage 3 Warning: ADMET step encountered issue: {str(e)}\\n"

    # -------------------------------------------------------------
    # Stage 4: Zip File & CSV Report Packaging
    # -------------------------------------------------------------
    status_log += "📦 Stage 4: Packaging CSV Reports & Docked Complexes ZIP Archive...\\n"
    progress(0.95, desc="Stage 4: Packaging Results")
    
    csv_file_path = "screening_results.csv"
    if admet_df is not None and not admet_df.empty:
        admet_df.to_csv(csv_file_path, index=False)
    elif filter_df is not None:
        filter_df.to_csv(csv_file_path, index=False)

    zip_file_path = "DeepDock_Results.zip"
    try:
        create_results_zip(zip_file_path, csv_file_path, docked_mols, filtered_names)
        status_log += "🎉 Pipeline Executed Successfully! Download your files below.\\n"
    except Exception:
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            if os.path.exists(csv_file_path):
                zipf.write(csv_file_path, arcname="screening_results.csv")
        status_log += "🎉 Pipeline Finished! Results zipped.\\n"

    progress(1.0, desc="Complete!")
    
    return status_log, filter_df, admet_df, csv_file_path, zip_file_path


# --- GRADIO INTERFACE LAYOUT ---
with gr.Blocks(title="DeepDock-AI Virtual Screening") as demo:
    gr.Markdown("# 🧬 DeepDock-AI: GPU-Accelerated Virtual Screening Pipeline")
    gr.Markdown("Integrated In-Silico Molecular Docking, RDKit Filters, and ADMETlab 3.0 Profiling")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Files & Screening Setup")
            ligand_input = gr.File(label="Upload Ligands (.sdf)", file_types=[".sdf"])
            protein_input = gr.File(label="Upload Target Protein (.pdb)", file_types=[".pdb"])
            
            filter_type = gr.Dropdown(
                choices=["Lipinski Rule of 5", "Veber Rule", "Pfizer Rule", "None"],
                value="Lipinski Rule of 5",
                label="Filter Criteria (Stage 1)"
            )
            
            batch_size = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="Docking Batch Size")
            use_admet_api = gr.Checkbox(value=True, label="Enable ADMETlab 3.0 Cloud API")
            
            run_btn = gr.Button("🚀 Run Full DeepDock Pipeline", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Live Console Status")
            console_output = gr.Textbox(label="Status & Execution Log", lines=8, interactive=False)

    gr.Markdown("---")
    
    with gr.Accordion("📋 Stage 1 Results: RDKit Molecular Descriptors & Filters", open=True):
        stage1_table = gr.Dataframe(label="Pre-docking Molecular Properties", interactive=False)

    with gr.Accordion("📊 Stage 2 & 3 Results: Docking Scores & ADMETlab 3.0 Profiling", open=True):
        admet_table = gr.Dataframe(label="Docking ΔG (kcal/mol) + ADMET Properties", interactive=False)

    gr.Markdown("### 📥 4. Download Outputs & Report Files")
    with gr.Row():
        csv_download = gr.File(label="📄 Download Screening Results (CSV)")
        zip_download = gr.File(label="📦 Download Complete Results Package (ZIP)")

    run_btn.click(
        fn=run_deepdock_pipeline,
        inputs=[ligand_input, protein_input, batch_size, filter_type, use_admet_api],
        outputs=[console_output, stage1_table, admet_table, csv_download, zip_download]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, show_error=True)
'''

with open("/kaggle/working/DeepDock--AI/app_gradio.py", "w", encoding="utf-8") as f:
    f.write(gradio_code)

print("✅ app_gradio.py clean syntax rewrite completed successfully!")
