import os
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
    
    # -------------------------------------------------------------
    # Stage 1: RDKit Pre-filtering & Descriptors
    # -------------------------------------------------------------
    status_log += "🔍 Stage 1: Running RDKit Molecular Filtering...\\n"
    progress(0.25, desc="Stage 1: RDKit Filters & Descriptors")
    
    try:
        filtered_mols, filtered_names, filtered_cids, filter_df = apply_rdkit_filters(
            ligand_file.name, filter_type=filter_type
        )
        status_log += f"✅ Stage 1 Complete: {len(filtered_mols)} molecules passed initial filters.\\n"
    except Exception as e:
        return f"❌ Stage 1 Error: {str(e)}", None, None, None, None

    if not filtered_mols:
        return "⚠️ No molecules passed the selected RDKit filters.", filter_df, None, None, None

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
        # Fallback Zip creation
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
        # LEFT COLUMN: Inputs & Controls
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

        # RIGHT COLUMN: Live Execution Log
        with gr.Column(scale=2):
            gr.Markdown("### Live Console Status")
            console_output = gr.Textbox(label="Status & Execution Log", lines=8, interactive=False)

    gr.Markdown("---")
    
    # STAGE 1 OUTPUT: RDKit Filtering Table
    with gr.Accordion("📋 Stage 1 Results: RDKit Molecular Descriptors & Filters", open=True):
        stage1_table = gr.Dataframe(label="Pre-docking Molecular Properties", interactive=False)

    # STAGE 3 OUTPUT: ADMET & Docking Table
    with gr.Accordion("📊 Stage 2 & 3 Results: Docking Scores & ADMETlab 3.0 Profiling", open=True):
        admet_table = gr.Dataframe(label="Docking ΔG (kcal/mol) + ADMET Properties", interactive=False)

    # DOWNLOADS SECTION
    gr.Markdown("### 📥 4. Download Outputs & Report Files")
    with gr.Row():
        csv_download = gr.File(label="📄 Download Screening Results (CSV)")
        zip_download = gr.File(label="📦 Download Complete Results Package (ZIP)")

    # Event Wiring
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

print("✅ Upgraded app_gradio.py with Stage 1 RDKit, Stage 3 ADMET, CSV & ZIP exports successfully!")
