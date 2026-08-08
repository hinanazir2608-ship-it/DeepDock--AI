import os
import pandas as pd
import gradio as gr
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
from modules.filters import apply_rdkit_filters
from modules.docking import run_batched_docking

# Define working directory safely inside DeepDock--AI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_admet_properties(mols, names):
    """Calculates ADMET descriptors using RDKit"""
    admet_data = []
    for idx, (mol, name) in enumerate(zip(mols, names)):
        if mol is None:
            continue
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        gi_absorption = "High" if tpsa < 140 and logp < 5 else "Low"
        bbb_permeable = "Yes" if tpsa < 90 and logp > 0 and logp < 3 else "No"
        
        admet_data.append({
            "CID": idx + 1,
            "Name": name,
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "Rotatable Bonds": rotatable_bonds,
            "HBD": hbd,
            "HBA": hba,
            "GI Absorption": gi_absorption,
            "BBB Permeable": bbb_permeable
        })
    return pd.DataFrame(admet_data)

def run_deepdock_pipeline(ligand_file, filter_type, target_pdbqt, progress=gr.Progress(track_tqdm=True)):
    status_log = ""

    if ligand_file is None:
        return "❌ Error: Please upload a ligand file.", None, None, None, None, None

    # Output file paths strictly inside working tree
    out_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(BASE_DIR, "docking_results.csv")
    zip_path = os.path.join(BASE_DIR, "docking_results.zip")

    # -------------------------------------------------------------
    # STAGE 1: RDKit Filtering
    # -------------------------------------------------------------
    progress(0.1, desc="🔍 Stage 1: RDKit Filtering")
    try:
        filtered_mols, filtered_names, filtered_cids, filter_df = apply_rdkit_filters(
            ligand_file.name, filter_type=filter_type
        )
        status_log += f"✅ Stage 1 Complete: {len(filtered_mols)} molecules passed initial filters.\n"
    except Exception as e:
        return f"❌ Stage 1 Error: {str(e)}", None, None, None, None, None

    if not filtered_mols or len(filtered_mols) == 0:
        return status_log + "\n⚠️ No molecules passed the selected RDKit filters.", filter_df, None, None, None, None

    # -------------------------------------------------------------
    # STAGE 2: Batched Vina Docking
    # -------------------------------------------------------------
    progress(0.3, desc=f"⚡ Stage 2: Running AutoDock Vina on {len(filtered_mols)} ligands...")
    try:
        docking_results = run_batched_docking(
            filtered_mols, target_pdbqt, grid_center=(0,0,0), grid_size=(20,20,20), 
            output_dir=out_dir, progress=progress
        )
        docking_df = pd.DataFrame(docking_results) if docking_results else pd.DataFrame()
        status_log += f"✅ Stage 2 Complete: Vina Docking executed for {len(filtered_mols)} molecules.\n"
    except Exception as e:
        return status_log + f"\n❌ Stage 2 Docking Error: {str(e)}", filter_df, None, None, None, None

    # -------------------------------------------------------------
    # STAGE 3: ADMET Analysis
    # -------------------------------------------------------------
    progress(0.8, desc="🧬 Stage 3: Calculating ADMET Descriptors")
    try:
        admet_df = calculate_admet_properties(filtered_mols, filtered_names)
        status_log += f"✅ Stage 3 Complete: ADMET profiles generated.\n"
    except Exception as e:
        status_log += f"⚠️ Stage 3 ADMET Warning: {str(e)}\n"
        admet_df = pd.DataFrame()

    # Save output CSV safely
    progress(0.95, desc="📊 Formatting Final Results & Downloads")
    if not docking_df.empty:
        docking_df.to_csv(csv_path, index=False)
    else:
        csv_path = None
        zip_path = None

    progress(1.0, desc="🎉 Pipeline Finished!")
    status_log += "🎉 DeepDock-AI Pipeline finished successfully!"
    
    return status_log, filter_df, docking_df, admet_df, csv_path, zip_path

# --- GRADIO INTERFACE BUILD ---
with gr.Blocks(title="DeepDock-AI") as demo:
    gr.Markdown("# 🧬 DeepDock-AI Pipeline")
    
    with gr.Row():
        ligand_input = gr.File(label="Upload Ligand File (.sdf, .csv)")
        filter_input = gr.Dropdown(["Lipinski", "None"], value="Lipinski", label="Filter Type")
        target_input = gr.File(label="Upload Target PDBQT")
        
    submit_btn = gr.Button("Run Pipeline", variant="primary")
    
    status_output = gr.Textbox(label="Status Logs", lines=6)
    
    with gr.Tabs():
        with gr.TabItem("1. Filtered Ligands"):
            filter_df_output = gr.DataFrame(label="Filtered Ligands Results")
        with gr.TabItem("2. Docking Results"):
            docking_df_output = gr.DataFrame(label="AutoDock Vina Scores")
        with gr.TabItem("3. ADMET Analysis"):
            admet_df_output = gr.DataFrame(label="ADMET Property Profiles")
            
    with gr.Row():
        csv_output = gr.File(label="Download Results CSV")
        zip_output = gr.File(label="Download Output ZIP")
    
    submit_btn.click(
        fn=run_deepdock_pipeline,
        inputs=[ligand_input, filter_input, target_input],
        outputs=[status_output, filter_df_output, docking_df_output, admet_df_output, csv_output, zip_output]
    )

if __name__ == "__main__":
    # Allowed paths added to grant Gradio access to /kaggle/working
    demo.launch(share=True, debug=True, allowed_paths=["/kaggle/working", BASE_DIR])