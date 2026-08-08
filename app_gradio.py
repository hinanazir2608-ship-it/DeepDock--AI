import os

# Ensure we are in the correct working directory
if os.path.exists("DeepDock--AI"):
    os.chdir("DeepDock--AI")

app_code = '''"""
DeepDock-AI — Gradio Front-End
"""

from __future__ import annotations
import gc
import os
import tempfile

import gradio as gr
import pandas as pd
import torch

from modules.preprocessing import load_ligands_from_bytes, parse_pdb_binding_site, parse_conf_txt
from modules.filters import run_parallel_admet_filter
from modules.docking import dock_ligands
from modules.admetlab import run_admet_analysis
from modules.export import export_pdb_complexes_zip, export_csv
from rdkit import Chem

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

def run_stage1(ligand_file, enable_admet, state, progress=gr.Progress()):
    if ligand_file is None:
        raise gr.Error("Upload a Ligands.sdf file first.")

    progress(0.1, desc="Loading ligands…")
    sdf_bytes = _read_bytes(ligand_file)
    raw_mols, raw_names = load_ligands_from_bytes(sdf_bytes)
    
    if len(raw_mols) == 0:
        raise gr.Error("No valid molecules found in that SDF file.")

    if enable_admet:
        progress(0.4, desc="Running parallel ADMET filter...")
        passed, records = run_parallel_admet_filter(raw_mols)
    else:
        passed, records = raw_mols, []

    progress(1.0, desc="Filter complete")

    state = dict(state or {})
    state.update(raw_mols=raw_mols, filtered_mols=passed, filter_records=records)

    summary = f"Input ligands: {len(raw_mols)}\\nPassed filter: {len(passed)}\\n✅ {len(passed)} ligands ready for Stage 2."
    filt_df = pd.DataFrame(records) if records else pd.DataFrame()
    filt_csv_path = _bytes_to_tempfile(export_csv(filt_df), "filter_report.csv") if records else None

    return summary, filt_df, filt_csv_path, state

def run_stage2_single(state, ligand_file, protein_file, conf_file, n_to_dock, exhaustiveness, n_poses, box_size_val, weights_file, progress=gr.Progress()):
    state = dict(state or {})
    mols_ready = state.get("filtered_mols") or state.get("raw_mols")

    # Fallback to direct SDF upload if state was reset
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

    state.update(docking_results=results)

    if not results:
        return "⚠️ Docking ran but produced no valid results.", pd.DataFrame(), None, state

    dock_records = [
        {
            "Rank": i + 1,
            "Compound CID / Name": getattr(r, 'name', f"Compound_{i+1}"),
            "Vina Score": round(getattr(r, 'vina_score', 0.0), 3),
            "GNINA ΔG (kcal/mol)": round(getattr(r, 'gnina_affinity', 0.0), 3),
        }
        for i, r in enumerate(results)
    ]
    dock_df = pd.DataFrame(dock_records)
    summary = f"Ligands docked: {len(results)}\\n✅ Docking complete. Move to Stage 3."

    complexes = [r.complex_pdb for r in results if hasattr(r, 'complex_pdb') and r.complex_pdb]
    zip_path = None
    if complexes:
        zip_bytes = export_pdb_complexes_zip(complexes, [r.name for r in results], [r.gnina_affinity for r in results])
        zip_path = _bytes_to_tempfile(zip_bytes, "docked_complexes.zip")

    return summary, dock_df, zip_path, state

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

        # Wiring
        run_filter_btn.click(
            fn=run_stage1, inputs=[ligand_input, enable_admet, state],
            outputs=[filter_summary, filter_table, filter_csv, state],
        )
        run_dock_btn.click(
            fn=run_stage2_single,
            inputs=[state, ligand_input, protein_input, conf_input, n_to_dock, exhaustiveness, n_poses, box_size_val, weights_input],
            outputs=[dock_summary, dock_table, dock_zip, state],
        )

    return demo
'''

with open("app_gradio.py", "w") as f:
    f.write(app_code)

print("✅ Clean app_gradio.py saved successfully without self-write code.")
