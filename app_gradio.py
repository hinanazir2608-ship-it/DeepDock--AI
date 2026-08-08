import os
import math
import pickle
import hashlib
import gc
import io
import time
from typing import List, Optional, Tuple
import pandas as pd
import torch
import gradio as gr

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Import modular components
from modules.preprocessing import load_ligands_from_bytes, parse_pdb_binding_site, parse_conf_txt, ensure_3d_coords
from modules.filters import run_parallel_admet_filter, compute_single_admet
from modules.docking import dock_ligands, load_gnina_model, VINA_AVAILABILITY
from modules.admetlab import run_admet_analysis
from modules.export import export_pdb_complexes_zip, export_csv, generate_docx_report
from modules.model import get_device_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_ligand_name(mol: Chem.Mol, index: int) -> str:
    if mol is None:
        return f"Compound_{index+1}"
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    elif mol.HasProp("PUBCHEM_COMPOUND_CID"):
        return f"CID_{mol.GetProp('PUBCHEM_COMPOUND_CID')}"
    elif mol.HasProp("PUBCHEM_IUPAC_NAME"):
        return mol.GetProp("PUBCHEM_IUPAC_NAME")
    elif mol.HasProp("ID"):
        return mol.GetProp("ID")
    return f"CID_{index+1}"

def run_pipeline(
    ligand_file, protein_file, conf_file, 
    enable_admet, filter_type, 
    exhaustiveness, n_poses, box_size_val, 
    use_admetlab_api, top_admet, weights_file,
    progress=gr.Progress(track_tqdm=True)
):
    status_log = ""
    
    if ligand_file is None:
        return "❌ Error: Please upload a Ligand library (.sdf).", None, None, None, None, None
    if protein_file is None:
        return "❌ Error: Please upload a Target Protein (.pdbqt).", None, None, None, None, None

    gc.collect()

    # -------------------------------------------------------------
    # STAGE 1: RDKit Filtering
    # -------------------------------------------------------------
    progress(0.1, desc="🔍 Stage 1: Loading & Filtering Ligands...")
    try:
        raw_mols, raw_names = load_ligands_from_bytes(ligand_file.read() if hasattr(ligand_file, 'read') else open(ligand_file.name, 'rb').read())
        raw_names = [raw_names[i] if (raw_names and i < len(raw_names) and raw_names[i]) else extract_ligand_name(m, i) for i, m in enumerate(raw_mols)]
        
        n_in = len(raw_mols)
        status_log += f"✅ Loaded {n_in} ligands.\n"

        if enable_admet:
            passed, records = run_parallel_admet_filter(raw_mols)
        else:
            passed, records = raw_mols, []
            
        filter_df = pd.DataFrame(records) if records else None
        status_log += f"✅ Stage 1 Complete: {len(passed)}/{n_in} molecules passed filters.\n"
    except Exception as e:
        return f"❌ Stage 1 Error: {str(e)}", None, None, None, None, None

    if not passed or len(passed) == 0:
        return status_log + "\n⚠️ No molecules passed the filters.", filter_df, None, None, None, None

    # -------------------------------------------------------------
    # STAGE 2: GNINA Molecular Docking
    # -------------------------------------------------------------
    progress(0.4, desc="⚡ Stage 2: Running GNINA Molecular Docking...")
    try:
        pdbqt_bytes = protein_file.read() if hasattr(protein_file, 'read') else open(protein_file.name, 'rb').read()
        site_info = parse_pdb_binding_site(pdbqt_bytes)
        
        conf_bytes = conf_file.read() if conf_file and hasattr(conf_file, 'read') else (open(conf_file.name, 'rb').read() if conf_file else None)
        grid_info = parse_conf_txt(conf_bytes) if conf_bytes else None

        if grid_info:
            center = grid_info["center"]
            box_size = grid_info["size"]
        else:
            center = site_info["centroid"]
            box_size = (box_size_val, box_size_val, box_size_val)

        weights_path = ""
        if weights_file is not None:
            tmp_weights = os.path.join(OUT_DIR, "gnina_weights.pt")
            with open(tmp_weights, "wb") as f:
                f.write(weights_file.read() if hasattr(weights_file, 'read') else open(weights_file.name, 'rb').read())
            weights_path = tmp_weights

        dock_names = [extract_ligand_name(m, i) for i, m in enumerate(passed)]
        
        results = dock_ligands(
            mols=passed,
            names=dock_names,
            pdb_bytes=pdbqt_bytes,
            center=center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            gnina_weights=weights_path
        )
        
        dock_records = []
        for i, r in enumerate(results):
            dock_records.append({
                "Rank": i + 1,
                "Compound CID / Name": r.name,
                "Vina Score": round(r.vina_score, 3),
                "GNINA ΔG (kcal/mol)": round(r.gnina_affinity, 3),
                "3D Pose": "✓" if getattr(r, 'pose_pdb', None) else "-"
            })
        docking_df = pd.DataFrame(dock_records)
        status_log += f"✅ Stage 2 Complete: Successfully docked {len(results)} ligands.\n"
    except Exception as e:
        return status_log + f"\n❌ Stage 2 Docking Error: {str(e)}", filter_df, None, None, None, None

    # -------------------------------------------------------------
    # STAGE 3: ADMET Analysis & Export Generation
    # -------------------------------------------------------------
    progress(0.8, desc="🧬 Stage 3: Running ADMET Profiling & Reports...")
    admet_df = pd.DataFrame()
    csv_path, zip_path, docx_path = None, None, None
    try:
        top_results = results[:top_admet]
        top_mols = [r.mol for r in top_results]
        top_names = [r.name for r in top_results]
        top_scores = [r.gnina_affinity for r in top_results]

        admet_df, admet_source = run_admet_analysis(
            mols=top_mols,
            names=top_names,
            scores=top_scores,
            use_api=use_admetlab_api
        )
        status_log += f"✅ Stage 3 Complete: ADMET generated using {admet_source}.\n"

        # File Exports
        csv_path = os.path.join(OUT_DIR, "docking_results.csv")
        docking_df.to_csv(csv_path, index=False)

        complex_pdbs = [r.complex_pdb for r in top_results if getattr(r, 'complex_pdb', None)]
        c_names = [r.name for r in top_results if getattr(r, 'complex_pdb', None)]
        c_scores = [r.gnina_affinity for r in top_results if getattr(r, 'complex_pdb', None)]
        
        if complex_pdbs:
            zip_bytes = export_pdb_complexes_zip(complex_pdbs, c_names, c_scores)
            zip_path = os.path.join(OUT_DIR, "top_hits_complexes.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)

        docx_bytes = generate_docx_report(
            filter_df=filter_df,
            docking_df=docking_df,
            admet_df=admet_df,
            admet_source=admet_source,
            n_input=n_in,
            n_filtered=len(passed),
            n_docked=len(results),
            protein_info=site_info,
            grid_info={"center": center, "size": box_size},
            top_n=top_admet
        )
        docx_path = os.path.join(OUT_DIR, "DeepDock_AI_Report.docx")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)

        status_log += "🎉 DeepDock-AI Pipeline finished successfully!"
    except Exception as e:
        status_log += f"⚠️ Stage 3 Export Warning: {str(e)}\n"

    progress(1.0, desc="Done!")
    return status_log, filter_df, docking_df, admet_df, csv_path, zip_path, docx_path

# --- GRADIO INTERFACE BUILD ---
with gr.Blocks(title="DeepDock-AI Virtual Screening Platform") as demo:
    gr.Markdown("# 🧬 DeepDock-AI: GPU-Accelerated Virtual Screening Platform")
    gr.Markdown("### 3-Stage Pipeline: RDKit Pre-filter ➔ GNINA Molecular Docking ➔ ADMETlab 3.0 Profiling")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Input Files")
            ligand_input = gr.File(label="Ligand Library (.sdf)", file_types=[".sdf"])
            protein_input = gr.File(label="Target Protein (.pdbqt)", file_types=[".pdbqt"])
            conf_input = gr.File(label="Grid Config File (.txt - Optional)", file_types=[".txt"])

            gr.Markdown("### ⚙️ Parameters")
            enable_admet = gr.Checkbox(value=True, label="Enable RDKit ADMET Pre-filter")
            filter_type = gr.Dropdown(["Lipinski", "None"], value="Lipinski", label="Filter Ruleset")
            
            exhaustiveness = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Vina Exhaustiveness")
            n_poses = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Poses per Ligand")
            box_size_val = gr.Slider(minimum=10, maximum=40, value=20, step=2, label="Grid Box Size (Å)")
            
            use_admetlab_api = gr.Checkbox(value=True, label="Use ADMETlab 3.0 API (Fallbacks to RDKit)")
            top_admet = gr.Slider(minimum=5, maximum=200, value=50, step=5, label="Top N for ADMET & Reports")
            weights_file = gr.File(label="GNINA Weights .pt (Optional)", file_types=[".pt", ".pth"])

            run_btn = gr.Button("🚀 Run Full Pipeline", variant="primary")

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Execution Status Logs", lines=6)
            
            with gr.Tabs():
                with gr.TabItem("1. Filtered Ligands"):
                    filter_df_output = gr.DataFrame(label="RDKit Filter Results")
                with gr.TabItem("2. GNINA Docking"):
                    docking_df_output = gr.DataFrame(label="AutoDock Vina & GNINA Scores")
                with gr.TabItem("3. ADMET Profiling"):
                    admet_df_output = gr.DataFrame(label="ADMET Property Profiles")

            gr.Markdown("### 📥 Download Results")
            with gr.Row():
                csv_output = gr.File(label="Download CSV")
                zip_output = gr.File(label="Download PDB Complexes ZIP")
                docx_output = gr.File(label="Download DOCX Report")

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            ligand_input, protein_input, conf_input,
            enable_admet, filter_type,
            exhaustiveness, n_poses, box_size_val,
            use_admetlab_api, top_admet, weights_file
        ],
        outputs=[
            status_output, filter_df_output, docking_df_output, 
            admet_df_output, csv_output, zip_output, docx_output
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True, allowed_paths=["/kaggle/working", BASE_DIR])
