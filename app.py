"""
DeepDock-AI — GPU-Accelerated Virtual Screening Platform
Gradio Interface (port 8000)
"""

from __future__ import annotations
import gc
import io
import time
import tempfile
import os
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd
import torch

from modules.preprocessing  import load_ligands_from_bytes, parse_pdb_binding_site, parse_conf_txt
from modules.filters        import run_parallel_admet_filter
from modules.model          import load_model, get_device_info
from modules.inference      import run_inference, rank_results
from modules.admet_export   import build_admet_dataframe, export_hits_sdf, generate_docx_report

from rdkit import Chem


# ── Load model once at startup ────────────────────────────────────────────────
_model, _device, _precision = load_model("")


def get_hw_info() -> str:
    info = get_device_info()
    if info["cuda_available"]:
        return (f"🟢 GPU: {info['gpu_name']}  |  VRAM: {info['vram_gb']} GB  "
                f"|  {info['precision']}  |  PyTorch {torch.__version__}")
    return f"🟡 CPU mode (no CUDA)  |  PyTorch {torch.__version__}"


# ═════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def run_screening(
    ligand_file,
    protein_file,
    conf_file,
    enable_admet: bool,
    top_n: int,
    batch_size: int,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, pd.DataFrame, str, str, str, str]:
    """
    Main pipeline. Returns:
      status_md, results_df, csv_path, parquet_path, sdf_path, docx_path
    """
    logs: List[str] = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    try:
        # ── 1. Load ligands ──────────────────────────────────────────────────
        progress(0.05, desc="Loading ligands…")
        if ligand_file is None:
            return "❌ Please upload a Ligands SDF file.", pd.DataFrame(), None, None, None, None

        with open(ligand_file.name, "rb") as f:
            sdf_bytes = f.read()

        all_mols, all_names = load_ligands_from_bytes(sdf_bytes)
        n_input = len(all_mols)
        if n_input == 0:
            return "❌ No valid molecules found in SDF file.", pd.DataFrame(), None, None, None, None
        log(f"✅ Loaded **{n_input}** ligands")

        # ── 2. Parse protein ─────────────────────────────────────────────────
        progress(0.10, desc="Parsing protein…")
        if protein_file is None:
            return "❌ Please upload a protein PDB file.", pd.DataFrame(), None, None, None, None

        with open(protein_file.name, "rb") as f:
            pdb_bytes = f.read()
        site_info = parse_pdb_binding_site(pdb_bytes)
        cx, cy, cz = site_info["centroid"]
        log(f"✅ Binding site centroid: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f} ({site_info['source']})")

        # ── 3. Grid config (optional) ────────────────────────────────────────
        grid_info = None
        if conf_file is not None:
            with open(conf_file.name, "rb") as f:
                conf_bytes = f.read()
            grid_info = parse_conf_txt(conf_bytes)
            gc, gs = grid_info["center"], grid_info["size"]
            log(f"✅ Grid: center=({gc[0]},{gc[1]},{gc[2]}) size=({gs[0]}×{gs[1]}×{gs[2]})")

        # ── 4. ADMET pre-filter ──────────────────────────────────────────────
        progress(0.20, desc="ADMET pre-filtering…")
        if enable_admet:
            t0 = time.time()
            passed_mols, _ = run_parallel_admet_filter(all_mols)
            n_passed = len(passed_mols)
            log(f"✅ ADMET filter: {n_passed}/{n_input} passed ({time.time()-t0:.1f}s)")
            if n_passed == 0:
                return (
                    "\n".join(logs) + "\n\n❌ No molecules passed ADMET filters. "
                    "Disable the filter or use a different ligand set.",
                    pd.DataFrame(), None, None, None, None,
                )
            mols_to_screen = passed_mols
        else:
            mols_to_screen = all_mols
            n_passed = n_input
            log("ℹ️ ADMET filter bypassed")

        # ── 5. GPU inference ─────────────────────────────────────────────────
        progress(0.35, desc="Running GNN inference…")
        t0 = time.time()

        # Simple progress callback using a list
        scored_mols, scores = run_inference(
            model       = _model,
            device      = _device,
            mols        = mols_to_screen,
            batch_size  = batch_size,
            use_fp16    = (_device.type == "cuda"),
        )
        elapsed = time.time() - t0
        log(f"✅ Scored {len(scored_mols)} ligands in {elapsed:.1f}s "
            f"({len(scored_mols)/max(elapsed,0.01):.0f} lig/sec)")

        # ── 6. Rank & ADMET profiling ────────────────────────────────────────
        progress(0.80, desc="Ranking & ADMET profiling…")
        top_mols, top_scores = rank_results(scored_mols, scores, top_n=top_n)
        top_names = [
            m.GetPropsAsDict().get("_Name", f"hit_{i+1}")
            for i, m in enumerate(top_mols)
        ]
        admet_df = build_admet_dataframe(top_mols, top_scores, top_names)
        log(f"✅ Top {len(top_mols)} hits ranked and profiled")

        # ── 7. Export files ──────────────────────────────────────────────────
        progress(0.90, desc="Generating export files…")

        tmp_dir = tempfile.mkdtemp()

        # CSV
        csv_path = os.path.join(tmp_dir, "deepdock_hits.csv")
        admet_df.to_csv(csv_path, index=False)

        # Parquet
        parquet_path = os.path.join(tmp_dir, "deepdock_hits.parquet")
        admet_df.to_parquet(parquet_path, index=False, engine="pyarrow")

        # SDF
        sdf_path = os.path.join(tmp_dir, "top_hits_docked.sdf")
        sdf_bytes = export_hits_sdf(top_mols, top_scores, top_names, admet_df)
        with open(sdf_path, "wb") as f:
            f.write(sdf_bytes)

        # DOCX
        docx_path = None
        try:
            docx_bytes = generate_docx_report(
                admet_df        = admet_df,
                n_input         = n_input,
                n_after_filter  = n_passed,
                top_n           = len(top_mols),
                protein_info    = site_info,
                grid_info       = grid_info,
            )
            docx_path = os.path.join(tmp_dir, "Screening_Report.docx")
            with open(docx_path, "wb") as f:
                f.write(docx_bytes)
            log("✅ DOCX report generated")
        except Exception as e:
            log(f"⚠️ DOCX skipped: {e}")

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Build status markdown
        best_dg = min(top_scores) if top_scores else float("nan")
        mean_qed = admet_df["QED"].mean() if "QED" in admet_df else float("nan")
        progress(1.0, desc="Done!")

        status_md = "\n".join(logs) + f"""

---
### 🏆 Screening Complete
| Metric | Value |
|--------|-------|
| Input ligands | {n_input} |
| Passed ADMET | {n_passed} |
| Scored | {len(scored_mols)} |
| Best ΔG (kcal/mol) | {best_dg:.3f} |
| Mean QED | {mean_qed:.3f} |
| Device | {str(_device).upper()} |
"""
        return status_md, admet_df, csv_path, parquet_path, sdf_path, docx_path

    except Exception as e:
        import traceback
        return f"❌ Error: {e}\n\n```\n{traceback.format_exc()}\n```", pd.DataFrame(), None, None, None, None


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="DeepDock-AI") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
# 🧬 DeepDock-AI — Virtual Screening Platform
**GPU-accelerated binding affinity prediction** · PyTorch GNN · RDKit · AMP FP16
""")

    hw_info = gr.Markdown(f"`{get_hw_info()}`")

    # ── Input Section ─────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📁 Input Files")
            with gr.Row():
                ligand_file  = gr.File(label="💊 Ligands.sdf", file_types=[".sdf"])
                protein_file = gr.File(label="🎯 protein.pdb", file_types=[".pdb"])
                conf_file    = gr.File(label="⚙️ conf.txt (optional)", file_types=[".txt", ".conf"])

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Parameters")
            enable_admet = gr.Checkbox(
                label="Enable 2D ADMET Pre-filter (Lipinski + Veber + PAINS)",
                value=True,
            )
            top_n = gr.Slider(
                label="Top N hits", minimum=5, maximum=200, value=50, step=5
            )
            batch_size = gr.Radio(
                label="GPU Batch Size",
                choices=[4, 8, 16, 32, 64],
                value=16,
            )

    # ── Run Button ────────────────────────────────────────────────────────────
    run_btn = gr.Button("▶ Start Virtual Screening", variant="primary", size="lg")

    # ── Output Section ────────────────────────────────────────────────────────
    gr.Markdown("### 📊 Results")

    with gr.Row():
        status_box = gr.Markdown("*Upload files and click Start to begin.*")

    results_table = gr.Dataframe(
        label="🏆 Top Hit Compounds — Ranked by ΔG (kcal/mol)",
        wrap=True,
        interactive=False,
    )

    gr.Markdown("### 📦 Download Results")
    with gr.Row():
        csv_out     = gr.File(label="⬇ CSV")
        parquet_out = gr.File(label="⬇ Parquet")
        sdf_out     = gr.File(label="⬇ SDF (3D poses)")
        docx_out    = gr.File(label="⬇ DOCX Report")

    # ── Sample Files ──────────────────────────────────────────────────────────
    with gr.Accordion("📂 Sample Test Files (click to expand)", open=False):
        gr.Markdown("""
Use these pre-built files to test the pipeline immediately:

| File | Description |
|------|-------------|
| `sample_ligands.sdf` | 19 real drug molecules (Aspirin, Ibuprofen, Erlotinib, Imatinib…) |
| `sample_protein.pdb` | SARS-CoV-2 Main Protease (3CLpro) simplified active site |
| `sample_conf.txt` | Grid center (10,5,20), box 22×22×22 Å |

Download from the file panel on the left, then upload above.
""")

    # ── About ────────────────────────────────────────────────────────────────
    with gr.Accordion("ℹ️ About DeepDock-AI", open=False):
        gr.Markdown("""
| Component | Technology |
|-----------|-----------|
| UI | Gradio |
| Docking model | PyTorch GCN + Attention Pooling GNN |
| GPU acceleration | CUDA FP16 AMP (Kaggle T4 / P100) |
| Cheminformatics | RDKit |
| Pre-filter | Lipinski RO5 + Veber + PAINS (multiprocessing) |
| ADMET | QED, TPSA, LogP, MW, ESOL LogS, BBB |
| Export | CSV · Parquet · SDF (3D) · DOCX report |

**OOM Recovery:** If CUDA runs out of memory, batch size is automatically halved and inference retries.
""")

    # ── Wire up ───────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_screening,
        inputs=[ligand_file, protein_file, conf_file, enable_admet, top_n, batch_size],
        outputs=[status_box, results_table, csv_out, parquet_out, sdf_out, docx_out],
    )

# ═════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    )
