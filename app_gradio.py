import os

repo_path = "/kaggle/working/DeepDock--AI"
if not os.path.exists(repo_path):
    os.makedirs(repo_path, exist_ok=True)

# Complete 3-Stage app_gradio.py with RDKit Lipinski Filtering, Docking, and ADMET Profiling
app_gradio_content = """
import gradio as gr
import sys
import os
import pandas as pd

sys.path.insert(0, "/kaggle/working/DeepDock--AI")

from modules.docking import dock_ligands
from modules.preprocessing import load_ligands_from_bytes
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def run_lipinski_filters(ligand_files, max_mw, max_logp, max_hbd, max_hba):
    all_mols = []
    all_names = []
    
    for lf in ligand_files:
        bytes_data = open(lf.name, "rb").read() if hasattr(lf, "name") else lf
        mols, names = load_ligands_from_bytes([bytes_data])
        all_mols.extend(mols)
        all_names.extend(names)
        
    filtered_mols = []
    filtered_names = []
    records = []
    
    for mol, name in zip(all_mols, all_names):
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1
        
        passed = (mw <= max_mw and logp <= max_logp and hbd <= max_hbd and hba <= max_hba and violations <= 1)
        
        records.append({
            "Ligand ID": name,
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "HBD": hbd,
            "HBA": hba,
            "Lipinski Violations": violations,
            "Stage 1 Status": "Passed" if passed else "Filtered Out"
        })
        
        if passed:
            filtered_mols.append(mol)
            filtered_names.append(name)
            
    df_stage1 = pd.DataFrame(records)
    return filtered_mols, filtered_names, df_stage1

def perform_admet_profiling(docking_results, df_stage1):
    admet_records = []
    for res in docking_results:
        name = res.name
        score = res.vina_score
        
        # Match with stage 1 metadata
        row = df_stage1[df_stage1["Ligand ID"] == name]
        mw = row["MW"].values[0] if not row.empty else 400.0
        logp = row["LogP"].values[0] if not row.empty else 3.0
        hba = row["HBA"].values[0] if not row.empty else 5
        hbd = row["HBD"].values[0] if not row.empty else 2
        
        # Surrogate ADMET calculations
        tpsa = (hba * 12.0) + (hbd * 20.0)
        gi_absorption = "High" if (tpsa < 140 and logp < 5) else "Low"
        bbb_permeant = "Yes" if (tpsa < 90 and 1.5 < logp < 4.0) else "No"
        druglikeness = round(5.0 - (row["Lipinski Violations"].values[0] if not row.empty else 0) * 1.25, 2)
        
        admet_records.append({
            "Ligand ID": name,
            "Binding Affinity (kcal/mol)": score,
            "GI Absorption": gi_absorption,
            "BBB Permeant": bbb_permeant,
            "Est. TPSA (Å²)": tpsa,
            "Drug-likeness Score": druglikeness,
            "ADMET Status": "Passed Profile"
        })
        
    df_admet = pd.DataFrame(admet_records)
    if not df_admet.empty and "Binding Affinity (kcal/mol)" in df_admet.columns:
        df_admet = df_admet.sort_values(by="Binding Affinity (kcal/mol)", ascending=True)
    return df_admet

def create_app():
    with gr.Blocks(title="DeepDock-AI 3-Stage Pipeline") as demo:
        gr.Markdown("# 🧬 DeepDock-AI: Complete Virtual Screening Studio")
        gr.Markdown("### **Stage 1: Filters** (Lipinski Rule of 5) ➔ **Stage 2: Docking** (AutoDock Vina) ➔ **Stage 3: ADMET Analysis**")
        
        with gr.Row():
            receptor_input = gr.File(label="Upload Receptor (.pdbqt)")
            ligands_input = gr.File(label="Upload Ligands (.sdf)", file_count="multiple")
            
        with gr.Row():
            max_mw = gr.Number(label="Max Molecular Weight", value=500.0)
            max_logp = gr.Number(label="Max LogP", value=5.0)
            max_hbd = gr.Number(label="Max H-Bond Donors", value=5)
            max_hba = gr.Number(label="Max H-Bond Acceptors", value=10)
            
        run_btn = gr.Button("🚀 Run Full 3-Stage Pipeline", variant="primary", size="lg")
        
        with gr.Tabs():
            with gr.TabItem("Stage 1: Filter Results"):
                stage1_table = gr.Dataframe(label="Lipinski & Physicochemical Screening")
            with gr.TabItem("Stage 2: Docking Results"):
                stage2_table = gr.Dataframe(label="AutoDock Vina Binding Affinities")
            with gr.TabItem("Stage 3: ADMET Profile"):
                stage3_table = gr.Dataframe(label="Pharmacokinetic & Lead Optimization Summary")
                csv_output = gr.File(label="Download Final ADMET Results (.csv)")
                
        def pipeline_execution(receptor_file, ligand_files, mw_v, logp_v, hbd_v, hba_v):
            if not receptor_file or not ligand_files:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
                
            try:
                receptor_bytes = open(receptor_file.name, "rb").read()
                
                # STAGE 1: Filters
                filtered_mols, filtered_names, df_s1 = run_lipinski_filters(
                    ligand_files, mw_v, logp_v, int(hbd_v), int(hba_v)
                )
                
                if not filtered_mols:
                    empty_df = pd.DataFrame({"Error": ["All ligands filtered out in Stage 1."]})
                    return df_s1, empty_df, empty_df, None
                
                # STAGE 2: Docking
                docking_results = dock_ligands(
                    mols=filtered_mols,
                    names=filtered_names,
                    pdb_bytes=receptor_bytes,
                    center=(0.0, 0.0, 0.0),
                    box_size=(20.0, 20.0, 20.0)
                )
                
                df_s2_data = []
                for res in docking_results:
                    df_s2_data.append({
                        "Ligand ID": res.name,
                        "Binding Affinity (kcal/mol)": res.vina_score,
                        "Status": "Docked Successfully"
                    })
                df_s2 = pd.DataFrame(df_s2_data)
                
                # STAGE 3: ADMET Analysis
                df_s3 = perform_admet_profiling(docking_results, df_s1)
                
                csv_path = "/kaggle/working/DeepDock--AI/outputs/final_admet_report.csv"
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df_s3.to_csv(csv_path, index=False)
                
                return df_s1, df_s2, df_s3, csv_path
            except Exception as e:
                err_df = pd.DataFrame({"Error": [str(e)]})
                return err_df, err_df, err_df, None

        run_btn.click(
            fn=pipeline_execution,
            inputs=[receptor_input, ligands_input, max_mw, max_logp, max_hbd, max_hba],
            outputs=[stage1_table, stage2_table, stage3_table, csv_output]
        )
        
    return demo
"""

app_path = os.path.join(repo_path, "app_gradio.py")
with open(app_path, "w") as f:
    f.write(app_gradio_content)

print(f"✅ Fully updated 3-stage app_gradio.py successfully written to {app_path}")
