import os

# PyTorch CUDA P100 error bypass
os.environ["CUDA_VISIBLE_DEVICES"] = ""

code_content = '''"""
DeepDock-AI — Gradio front-end (Fixed State & Direct Fallback)
"""

from __future__ import annotations
import gc
import os
import pickle
import hashlib
import math
import tempfile
from typing import Optional

import gradio as gr
import pandas as pd
import torch

from modules.preprocessing import (
    load_ligands_from_bytes,
    parse_pdb_binding_site,
    parse_conf_txt,
)
from modules.filters import run_parallel_admet_filter
from modules.docking import dock_ligands, VINA_AVAILABLE
from modules.admetlab import run_admet_analysis
from modules.export import export_pdb_complexes_zip, export_csv, generate_docx_report
from modules.model import get_device_info
from rdkit import Chem

CHECKPOINT_DIR = "batch_checkpoints"


def extract_ligand_name(mol: Chem.Mol, index: int) -> str:
    if mol is None:
        return f"Compound_{index+1}"
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    elif mol.HasProp("PUBCHEM_COMPOUND_CID"):
        return f"CID_{mol.GetProp(\'PUBCHEM_COMPOUND_CID\')}"
    elif mol.HasProp("PUBCHEM_IUPAC_NAME"):
        return mol.GetProp("PUBCHEM_IUPAC_NAME")
    elif mol.HasProp("ID"):
        return mol.GetProp("ID")
    return f"CID_{index+1}"


def _read_bytes(file_obj) -> bytes:
    if file_obj is None:
        return b""
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path or not os.path.exists(path):
        raise gr.Error("Could not read the uploaded file.")
    with open(path, "rb") as f:
        return f.read()


def _bytes_to_tempfile(data: bytes, filename: str) -> str:
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _mols_to_temp_sdf(mols: list[Chem.Mol]) -> list[str]:
    paths = []
    for i, m in enumerate(mols):
        p = os.path.join(tempfile.gettempdir(), f"ligand_{i}.sdf")
        writer = Chem.SDWriter(p)
        writer.write(m)
        writer.close()
        paths.append(p)
    return paths


# STAGE 1 — RDKit ADMET Pre-Filter
def run_stage1(ligand_file, enable_admet, state, progress=gr.Progress()):
    if ligand_file is None:
        raise gr.Error("Upload a Ligands.sdf file first.")

    progress(0.1, desc="Loading ligands…")
    sdf_bytes = _read_bytes(ligand_file)
    raw_mols, raw_names = load_ligands_from_bytes(sdf_bytes)
    raw_names = [
        raw_names[i] if (raw_names and i < len(raw_names) and raw_names[i])
        else extract_ligand_name(m, i)
        for i, m in enumerate(raw_mols)
    ]
    n_in = len(raw_mols)
    if n_in == 0:
        raise gr.Error("No valid molecules found in that SDF file.")

    if enable_admet:
        progress(0.4, desc="Running parallel ADMET filter...")
        passed, records = run_parallel_admet_filter(raw_mols)
    else:
        passed, records = raw_mols, []

    progress(1.0, desc="Filter complete")

    state = dict(state or {})
    state.update(
        raw_mols=raw_mols, raw_names=raw_names,
        filtered_mols=passed, filter_records=records,
    )

    n_pass = len(passed)
    summary = (
        f"Input ligands: {n_in}\\n"
        f"Passed filter: {n_pass}\\n"
        f"Rejected: {n_in - n_pass}\\n"
        f"Pass rate: {n_pass / max(n_in, 1) * 100:.1f}%\\n"
    )
    if n_pass:
        summary += f"\\n✅ {n_pass} ligands ready for Stage 2."

    filt_df = pd.DataFrame(records) if records else pd.DataFrame()
    filt_csv_path = _bytes_to_tempfile(export_csv(filt_df), "filter_report.csv") if records else None

    return summary, filt_df, filt_csv_path, state


# STAGE 2 — Single-Shot Docking
def run_stage2_single(
    state, ligand_file, protein_file, conf_file, n_to_dock,
    exhaustiveness, n_poses, box_size_val, weights_file,
    progress=gr.Progress(),
):
    state = dict(state or {})
    mols_ready = state.get("filtered_mols") or state.get("raw_mols")

    # Fallback: Load directly from uploaded ligand file if state lost
    if not mols_ready and ligand_file is not None:
        sdf_bytes = _read_bytes(ligand_file)
        mols_ready, _ = load_ligands_from_bytes(sdf_bytes)

    if not mols_ready:
        raise gr.Error("No ligands found! Please upload Ligands.sdf in Stage 1 or Stage 2.")

    if protein_file is None:
        raise gr.Error("Upload protein.pdbqt first.")

    n_to_dock = max(1, min(int(n_to_dock), len(mols_ready)))
    dock_mols = mols_ready[:n_to_dock]
    ligand_paths = _mols_to_temp_sdf(dock_mols)

    pdbqt_bytes = _read_bytes(protein_file)
    site_info = parse_pdb_binding_site(pdbqt_bytes)
    grid_info = parse_conf_txt(_read_bytes(conf_file)) if conf_file else None

    if grid_info:
        center, box_size = grid_info["center"], grid_info["size"]
    else:
        center = site_info["centroid"]
        box_size = [box_size_val, box_size_val, box_size_val]

    receptor_path = _bytes_to_tempfile(pdbqt_bytes, "receptor.pdbqt")

    results = dock_ligands(
        receptor_path=receptor_path,
        ligand_paths=ligand_paths,
        center=center,
        box_size=box_size,
        exhaustiveness=int(exhaustiveness),
        n_poses=int(n_poses)
    )

    state.update(docking_results=results, protein_info=site_info, grid_info=grid_info)

    if not results:
        return "⚠️ Docking ran but produced no valid results.", pd.DataFrame(), None, state

    dock_records = [
        {
            "Rank": i + 1,
            "Compound CID / Name": getattr(r, 'name', f"Compound_{i+1}"),
            "Vina Score": round(getattr(r, 'vina_score', 0.0), 3),
            "GNINA ΔG (kcal/mol)": round(getattr(r, 'gnina_affinity', 0.0), 3),
            "3D Pose": "✓" if getattr(r, 'pose_pdb', None) else "–",
        }
        for i, r in enumerate(results)
    ]
    dock_df = pd.DataFrame(dock_records)
    summary = f"Ligands docked: {len(results)}\\n✅ Docking complete. Move to Stage 3."

    complexes = [r.complex_pdb for r in results if hasattr(r, 'complex_pdb') and r.complex_pdb]
    names_dock = [r.name for r in results if hasattr(r, 'complex_pdb') and r.complex_pdb]
    scores_dock = [r.gnina_affinity for r in results if hasattr(r, 'complex_pdb') and r.complex_pdb]
    zip_path = None
    if complexes:
        zip_bytes = export_pdb_complexes_zip(complexes, names_dock, scores_dock)
        zip_path = _bytes_to_tempfile(zip_bytes, "docked_complexes.zip")

    return summary, dock_df, zip_path, state


# STAGE 3 — ADMET Analysis
def run_stage3(state, top_admet, use_admetlab_api, progress=gr.Progress()):
    state = dict(state or {})
    docking_results = state.get("docking_results")
    if not docking_results:
        raise gr.Error("Run Stage 2 (docking) first.")

    top_admet = int(top_admet)
    top_results = docking_results[:top_admet]
    top_mols = [r.mol for r in top_results if hasattr(r, 'mol')]
    top_names = [r.name for r in top_results if hasattr(r, 'name')]
    top_scores = [r.gnina_affinity for r in top_results if hasattr(r, 'gnina_affinity')]

    progress(0.3, desc="Running ADMET analysis…")
    admet_df, source = run_admet_analysis(
        mols=top_mols, names=top_names, scores=top_scores,
        use_api=use_admetlab_api
    )

    state.update(admet_df=admet_df, admet_source=source)
    csv_path = _bytes_to_tempfile(export_csv(admet_df), "admet_results.csv")

    summary = f"Compounds analysed: {len(admet_df)}\\nADMET source: {source}"
    return summary, admet_df, csv_path, state


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DeepDock-AI") as demo:
        gr.Markdown("# 🧬 DeepDock-AI")

        state = gr.State({})

        with gr.Tab("🔬 Stage 1 — RDKit Filter"):
            with gr.Row():
                ligand_input = gr.File(label="Ligands.sdf", file_types=[".sdf"])
                enable_admet = gr.Checkbox(label="Enable ADMET pre-filter", value=True)
            run_filter_btn = gr.Button("▶ Run RDKit Filter", variant="primary")
            filter_summary = gr.Textbox(label="Summary", lines=4, interactive=False)
            filter_table = gr.Dataframe(label="Filter Report", wrap=True)
            filter_csv = gr.File(label="⬇ Filter Report CSV")

        with gr.Tab("⚗️ Stage 2 — Docking"):
            with gr.Row():
                protein_input = gr.File(label="protein.pdbqt", file_types=[".pdbqt"])
                conf_input = gr.File(label="conf.txt (optional)", file_types=[".txt", ".conf"])
                weights_input = gr.File(label="GNINA weights .pt (optional)", file_types=[".pt", ".pth"])
            with gr.Row():
                exhaustiveness = gr.Slider(1, 32, value=8, step=1, label="Vina exhaustiveness")
                n_poses = gr.Slider(1, 20, value=5, step=1, label="Poses per ligand")
                box_size_val = gr.Slider(10, 40, value=20, step=1, label="Grid box size (Å)")

            n_to_dock = gr.Number(label="Ligands to dock", value=50, precision=0)
            run_dock_btn = gr.Button("▶ Run Docking", variant="primary")
            dock_summary = gr.Textbox(label="Summary", lines=4, interactive=False)
            dock_table = gr.Dataframe(label="Docking Results", wrap=True)
            dock_zip = gr.File(label="⬇ PDB Complexes (ZIP)")

        with gr.Tab("📊 Stage 3 — ADMET & Export"):
            with gr.Row():
                top_admet = gr.Slider(5, 200, value=50, step=1, label="Top N for ADMET")
                use_admetlab_api = gr.Checkbox(label="Use ADMETlab 3.0 API", value=True)
            run_admet_btn = gr.Button("▶ Run ADMET Analysis", variant="primary")
            admet_summary = gr.Textbox(label="Summary", lines=3, interactive=False)
            admet_table = gr.Dataframe(label="ADMET Profiles", wrap=True)
            admet_csv = gr.File(label="⬇ ADMET CSV")

        # Wiring
        run_filter_btn.click(
            fn=run_stage1, inputs=[ligand_input, enable_admet, state],
            outputs=[filter_summary, filter_table, filter_csv, state],
        )
        run_dock_btn.click(
            fn=run_stage2_single,
            inputs=[state, ligand_input, protein_input, conf_input, n_to_dock,
                    exhaustiveness, n_poses, box_size_val, weights_input],
            outputs=[dock_summary, dock_table, dock_zip, state],
        )
        run_admet_btn.click(
            fn=run_stage3, inputs=[state, top_admet, use_admetlab_api],
            outputs=[admet_summary, admet_table, admet_csv, state],
        )

    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True, server_port=7880)
'''

# File Save Karein
with open("DeepDock--AI/app_gradio.py", "w") as f:
    f.write(code_content)

print("✅ File successfully overwritten with fallback mechanism.")

# Launch Fresh App
import sys
import importlib

if not os.getcwd().endswith("DeepDock--AI"):
    os.chdir("./DeepDock--AI")
sys.path.append(".")

import app_gradio
importlib.reload(app_gradio)

app_gradio.build_app().launch(share=True, inline=False, server_port=7880)
