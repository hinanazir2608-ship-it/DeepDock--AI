"""
DeepDock-AI — Gradio front-end
Same 3-stage pipeline as app.py (Streamlit): RDKit Filter → Vina/GNINA Docking → ADMETlab 3.0.

Reuses modules/preprocessing.py, filters.py, docking.py, admetlab.py, export.py,
model.py UNCHANGED. Those modules were written against Streamlit's st.progress()
(.progress(fraction)) and st.empty() (.text()/.info()/.success()/...) objects —
the two small adapter classes below (_ProgressAdapter, _StatusLogger) give
Gradio's progress tracker + a plain log list the same interface, so nothing in
modules/ had to change.
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


# ── Helper: PubChem CID / name extraction (identical to app.py) ───────────
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


# ── Adapters: let Streamlit-shaped modules run under Gradio unmodified ────
class _ProgressAdapter:
    """Mimics st.progress()'s .progress(fraction) — forwards to gr.Progress()."""

    def __init__(self, gr_progress_fn, desc: str = ""):
        self._fn = gr_progress_fn
        self._desc = desc

    def progress(self, fraction, text: Optional[str] = None):
        try:
            self._fn(fraction, desc=text or self._desc)
        except Exception:
            pass


class _StatusLogger:
    """Mimics st.empty()'s .text()/.info()/.success()/.warning()/.error() —
    accumulates lines instead of live-updating a single widget, since Gradio
    doesn't have a single-line mutable text placeholder the same way."""

    def __init__(self):
        self.lines: list[str] = []

    def text(self, msg):
        self.lines.append(str(msg))

    def info(self, msg):
        self.lines.append(f"ℹ️ {msg}")

    def success(self, msg):
        self.lines.append(f"✅ {msg}")

    def warning(self, msg):
        self.lines.append(f"⚠️ {msg}")

    def error(self, msg):
        self.lines.append(f"❌ {msg}")

    def dump(self, max_lines: int = 40) -> str:
        return "\n".join(self.lines[-max_lines:])


def _read_bytes(file_obj) -> bytes:
    """Gradio's gr.File (type='filepath') hands back a plain path string;
    older Gradio builds sometimes wrap it in an object with .name — handle both."""
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


# ═════════════════════════════════════════════════════════════════════════
#  STAGE 1 — RDKit ADMET Pre-Filter
# ═════════════════════════════════════════════════════════════════════════
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
        progress(0.4, desc="Running parallel ADMET filter (Lipinski + Veber + PAINS)…")
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
        f"Input ligands: {n_in}\n"
        f"Passed filter: {n_pass}\n"
        f"Rejected: {n_in - n_pass}\n"
        f"Pass rate: {n_pass / max(n_in, 1) * 100:.1f}%\n"
    )
    if n_in <= 20 and enable_admet:
        summary += "\n⚠️ Small set — consider disabling the ADMET pre-filter."
    if n_pass:
        summary += f"\n✅ {n_pass} ligands ready. Move to Stage 2."

    filt_df = pd.DataFrame(records) if records else pd.DataFrame()
    filt_csv_path = _bytes_to_tempfile(export_csv(filt_df), "filter_report.csv") if records else None

    return summary, filt_df, filt_csv_path, state


# ═════════════════════════════════════════════════════════════════════════
#  STAGE 2 — Vina / GNINA Docking (single-shot mode)
# ═════════════════════════════════════════════════════════════════════════
def run_stage2_single(
    state, protein_file, conf_file, n_to_dock,
    exhaustiveness, n_poses, box_size_val, weights_file,
    progress=gr.Progress(),
):
    state = dict(state or {})
    mols_ready = state.get("filtered_mols")
    if not mols_ready:
        raise gr.Error("Run Stage 1 (filter) first.")
    if protein_file is None:
        raise gr.Error("Upload protein.pdbqt first.")

    n_to_dock = max(1, min(int(n_to_dock), len(mols_ready)))
    dock_mols = mols_ready[:n_to_dock]
    dock_names = [extract_ligand_name(m, i) for i, m in enumerate(dock_mols)]

    pdbqt_bytes = _read_bytes(protein_file)
    site_info = parse_pdb_binding_site(pdbqt_bytes)
    grid_info = parse_conf_txt(_read_bytes(conf_file)) if conf_file else None

    if grid_info:
        center, box_size = grid_info["center"], grid_info["size"]
    else:
        center = site_info["centroid"]
        box_size = (box_size_val, box_size_val, box_size_val)

    weights_path = ""
    if weights_file is not None:
        weights_bytes = _read_bytes(weights_file)
        weights_path = _bytes_to_tempfile(weights_bytes, "gnina_weights.pt")

    prog_adapter = _ProgressAdapter(progress, desc="Docking…")
    status_log = _StatusLogger()

    results = dock_ligands(
        mols=dock_mols, names=dock_names, pdb_bytes=pdbqt_bytes,
        center=center, box_size=box_size,
        exhaustiveness=int(exhaustiveness), n_poses=int(n_poses),
        progress_bar=prog_adapter, status_text=status_log,
        gnina_weights=weights_path,
    )

    state.update(docking_results=results, protein_info=site_info, grid_info=grid_info)

    if not results:
        return (
            "⚠️ Docking ran but produced no valid results. "
            "Check grid center/box size or the protein file.",
            pd.DataFrame(), None, state,
        )

    dock_records = [
        {
            "Rank": i + 1, "Compound CID / Name": r.name,
            "Vina Score": round(r.vina_score, 3),
            "GNINA ΔG (kcal/mol)": round(r.gnina_affinity, 3),
            "3D Pose": "✓" if r.pose_pdb else "–",
        }
        for i, r in enumerate(results)
    ]
    dock_df = pd.DataFrame(dock_records)

    best = min(r.gnina_affinity for r in results)
    mean = sum(r.gnina_affinity for r in results) / len(results)
    summary = (
        f"Ligands docked: {len(results)}\n"
        f"Best ΔG: {best:.3f} kcal/mol\n"
        f"Mean ΔG: {mean:.3f} kcal/mol\n"
        f"3D poses generated: {sum(1 for r in results if r.pose_pdb)}\n"
        f"✅ Docking complete. Move to Stage 3."
    )

    complexes = [r.complex_pdb for r in results if r.complex_pdb]
    names_dock = [r.name for r in results if r.complex_pdb]
    scores_dock = [r.gnina_affinity for r in results if r.complex_pdb]
    zip_path = None
    if complexes:
        zip_bytes = export_pdb_complexes_zip(complexes, names_dock, scores_dock)
        zip_path = _bytes_to_tempfile(zip_bytes, "docked_complexes.zip")

    return summary, dock_df, zip_path, state


# ═════════════════════════════════════════════════════════════════════════
#  STAGE 2b — Batch docking (large libraries, elite-pool funnel)
#  Auto-resumes from an on-disk checkpoint if one matches the current
#  ligand set + protein + batch settings (same crash-recovery idea as the
#  Streamlit version's Resume button, just automatic instead of a
#  separate control — simpler UI, same safety net).
# ═════════════════════════════════════════════════════════════════════════
def run_stage2_batch(
    state, protein_file, conf_file, batch_size, top_k_per_batch,
    final_exhaustiveness, exhaustiveness, n_poses, box_size_val, weights_file,
    progress=gr.Progress(),
):
    state = dict(state or {})
    mols_ready = state.get("filtered_mols")
    if not mols_ready:
        raise gr.Error("Run Stage 1 (filter) first.")
    if protein_file is None:
        raise gr.Error("Upload protein.pdbqt first.")

    batch_size, top_k_per_batch = int(batch_size), int(top_k_per_batch)
    final_exhaustiveness, exhaustiveness, n_poses = (
        int(final_exhaustiveness), int(exhaustiveness), int(n_poses),
    )

    pdbqt_bytes = _read_bytes(protein_file)
    site_info = parse_pdb_binding_site(pdbqt_bytes)
    grid_info = parse_conf_txt(_read_bytes(conf_file)) if conf_file else None
    if grid_info:
        center, box_size = grid_info["center"], grid_info["size"]
    else:
        center = site_info["centroid"]
        box_size = (box_size_val, box_size_val, box_size_val)

    weights_path = ""
    if weights_file is not None:
        weights_path = _bytes_to_tempfile(_read_bytes(weights_file), "gnina_weights.pt")

    n_total = len(mols_ready)
    n_batches = math.ceil(n_total / batch_size)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ligand_key_src = "".join(extract_ligand_name(m, i) for i, m in enumerate(mols_ready[:50]))
    ckpt_key = hashlib.md5(
        (ligand_key_src + pdbqt_bytes[:5000].hex()
         + f"{batch_size}-{top_k_per_batch}").encode()
    ).hexdigest()[:12]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ckpt_{ckpt_key}.pkl")

    elite_pool, start_batch = [], 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        elite_pool = ckpt["elite_pool"]
        start_batch = ckpt["completed_batches"]
        yield (
            f"⚠️ Resuming from checkpoint: {start_batch}/{n_batches} batches "
            f"already done ({len(elite_pool)} elite ligands saved).",
            pd.DataFrame(), None, state,
        )

    for b in range(start_batch, n_batches):
        bstart, bend = b * batch_size, min((b + 1) * batch_size, n_total)
        batch_mols = mols_ready[bstart:bend]
        batch_names = [extract_ligand_name(m, bstart + i) for i, m in enumerate(batch_mols)]

        prog_adapter = _ProgressAdapter(
            progress, desc=f"Batch {b + 1}/{n_batches} ({bstart + 1}-{bend})"
        )
        status_log = _StatusLogger()

        try:
            batch_results = dock_ligands(
                mols=batch_mols, names=batch_names, pdb_bytes=pdbqt_bytes,
                center=center, box_size=box_size,
                exhaustiveness=exhaustiveness, n_poses=n_poses,
                progress_bar=prog_adapter, status_text=status_log,
                gnina_weights=weights_path,
            )
        except Exception as e:
            yield (
                f"❌ Batch {b + 1} crashed: {e}\n\n"
                f"Batches 1–{b} are saved in the checkpoint — re-run this "
                f"button and it will resume automatically from batch {b + 1}.",
                pd.DataFrame(), None, state,
            )
            return

        top_batch = batch_results[:top_k_per_batch]
        elite_pool.extend(top_batch)

        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {"completed_batches": b + 1, "elite_pool": elite_pool, "n_batches": n_batches}, f,
            )

        yield (
            f"✅ Batch {b + 1}/{n_batches} done — "
            f"{len(batch_results)}/{len(batch_mols)} docked, "
            f"kept top {len(top_batch)} → elite pool now {len(elite_pool)}.",
            pd.DataFrame(), None, state,
        )

    # ── Final consolidation round on the elite pool ──────────────────────
    prog_adapter = _ProgressAdapter(progress, desc="Final consolidation round…")
    status_log = _StatusLogger()
    elite_mols = [r.mol for r in elite_pool]
    elite_names = [r.name for r in elite_pool]

    try:
        final_results = dock_ligands(
            mols=elite_mols, names=elite_names, pdb_bytes=pdbqt_bytes,
            center=center, box_size=box_size,
            exhaustiveness=final_exhaustiveness, n_poses=n_poses,
            progress_bar=prog_adapter, status_text=status_log,
            gnina_weights=weights_path,
        )
    except Exception as e:
        yield (
            f"❌ Final consolidation round crashed: {e}\n\n"
            f"All {len(elite_pool)} batches are safe in the checkpoint — "
            f"re-run this button to retry just the final round.",
            pd.DataFrame(), None, state,
        )
        return

    state.update(docking_results=final_results, protein_info=site_info, grid_info=grid_info)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    dock_df = pd.DataFrame([
        {
            "Rank": i + 1, "Compound CID / Name": r.name,
            "Vina Score": round(r.vina_score, 3),
            "GNINA ΔG (kcal/mol)": round(r.gnina_affinity, 3),
            "3D Pose": "✓" if r.pose_pdb else "–",
        }
        for i, r in enumerate(final_results)
    ])

    complexes = [r.complex_pdb for r in final_results if r.complex_pdb]
    names_dock = [r.name for r in final_results if r.complex_pdb]
    scores_dock = [r.gnina_affinity for r in final_results if r.complex_pdb]
    zip_path = None
    if complexes:
        zip_path = _bytes_to_tempfile(
            export_pdb_complexes_zip(complexes, names_dock, scores_dock), "docked_complexes.zip"
        )

    yield (
        f"🎉 Batched docking complete! {len(final_results)} final ligands ranked. Move to Stage 3.",
        dock_df, zip_path, state,
    )


# ═════════════════════════════════════════════════════════════════════════
#  STAGE 3 — ADMET Analysis & Export
# ═════════════════════════════════════════════════════════════════════════
def run_stage3(state, top_admet, use_admetlab_api, progress=gr.Progress()):
    state = dict(state or {})
    docking_results = state.get("docking_results")
    if not docking_results:
        raise gr.Error("Run Stage 2 (docking) first.")

    top_admet = int(top_admet)
    top_results = docking_results[:top_admet]
    top_mols = [r.mol for r in top_results]
    top_names = [r.name for r in top_results]
    top_scores = [r.gnina_affinity for r in top_results]

    status_log = _StatusLogger()
    progress(0.3, desc="Running ADMET analysis…")
    admet_df, source = run_admet_analysis(
        mols=top_mols, names=top_names, scores=top_scores,
        use_api=use_admetlab_api, status_text=status_log,
    )
    progress(0.9, desc="Building exports…")

    state.update(admet_df=admet_df, admet_source=source)

    csv_path = _bytes_to_tempfile(export_csv(admet_df), "admet_results.csv")

    top_dock = docking_results[:top_admet]
    complexes = [r.complex_pdb for r in top_dock if r.complex_pdb]
    c_names = [r.name for r in top_dock if r.complex_pdb]
    c_scores = [r.gnina_affinity for r in top_dock if r.complex_pdb]
    zip_path = None
    if complexes:
        zip_path = _bytes_to_tempfile(
            export_pdb_complexes_zip(complexes, c_names, c_scores), "top_hits_complexes.zip"
        )

    dock_summary_df = pd.DataFrame([
        {"Rank": i + 1, "Name": r.name, "Vina Score": round(r.vina_score, 3),
         "GNINA ΔG (kcal/mol)": round(r.gnina_affinity, 3)}
        for i, r in enumerate(top_dock)
    ])
    filter_df_out = pd.DataFrame(state["filter_records"]) if state.get("filter_records") else None

    docx_path = None
    try:
        docx_bytes = generate_docx_report(
            filter_df=filter_df_out, docking_df=dock_summary_df, admet_df=admet_df,
            admet_source=source,
            n_input=len(state.get("raw_mols", [])),
            n_filtered=len(state.get("filtered_mols", [])),
            n_docked=len(docking_results),
            protein_info=state.get("protein_info"), grid_info=state.get("grid_info"),
            top_n=top_admet,
        )
        docx_path = _bytes_to_tempfile(docx_bytes, "DeepDock_AI_Report.docx")
    except Exception as e:
        gr.Warning(f"DOCX report failed to generate: {e}")

    progress(1.0, desc="Done")
    summary = f"Compounds analysed: {len(admet_df)}\nADMET source: {source}"
    return summary, admet_df, csv_path, zip_path, docx_path, state


# ═════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════
def build_app() -> gr.Blocks:
    dev = get_device_info()
    hw_line = (
        f"🟢 {dev['gpu_name']} · VRAM {dev['vram_gb']} GB · {dev['precision']}"
        if dev["cuda_available"] else "🟡 CPU mode — docking will be slower"
    )
    vina_line = (
        "✅ AutoDock-Vina ready" if VINA_AVAILABLE else
        "❌ vina/obabel not found — install with `pip install vina meeko` "
        "and `apt install openbabel`"
    )

    with gr.Blocks(title="DeepDock-AI") as demo:
        gr.Markdown("# 🧬 DeepDock-AI")
        gr.Markdown(
            "**3-Stage Pipeline:** 🔬 RDKit ADMET Filter → "
            "⚗️ AutoDock-Vina / GNINA Docking → 📊 ADMETlab 3.0 Profiling"
        )
        gr.Markdown(f"{hw_line}  \nPyTorch {torch.__version__}  \n{vina_line}")

        state = gr.State({})

        with gr.Tab("🔬 Stage 1 — RDKit Filter"):
            with gr.Row():
                ligand_input = gr.File(label="Ligands.sdf", file_types=[".sdf"])
                enable_admet = gr.Checkbox(label="Enable ADMET pre-filter", value=True)
            run_filter_btn = gr.Button("▶ Run RDKit Filter", variant="primary")
            filter_summary = gr.Textbox(label="Summary", lines=5, interactive=False)
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

            with gr.Tab("Single-shot"):
                n_to_dock = gr.Number(label="Ligands to dock (top N from filtered set)", value=50, precision=0)
                run_dock_btn = gr.Button("▶ Run Docking", variant="primary")
                dock_summary = gr.Textbox(label="Summary", lines=5, interactive=False)
                dock_table = gr.Dataframe(label="Docking Results", wrap=True)
                dock_zip = gr.File(label="⬇ PDB Complexes (ZIP)")

            with gr.Tab("Batch mode (large libraries)"):
                gr.Markdown(
                    "Splits the filtered set into batches, keeps the top hits per "
                    "batch, then re-docks the combined elite pool for a final "
                    "ranking. Auto-resumes from a checkpoint if this run crashes "
                    "partway through."
                )
                with gr.Row():
                    batch_size = gr.Number(label="Batch size", value=5000, precision=0)
                    top_k_per_batch = gr.Number(label="Keep top-K per batch", value=10, precision=0)
                    final_exhaustiveness = gr.Slider(1, 64, value=16, step=1, label="Final round exhaustiveness")
                run_batch_btn = gr.Button("▶ Run Batched Docking", variant="primary")
                batch_summary = gr.Textbox(label="Progress / Summary", lines=6, interactive=False)
                batch_table = gr.Dataframe(label="Final Docking Results", wrap=True)
                batch_zip = gr.File(label="⬇ PDB Complexes (ZIP)")

        with gr.Tab("📊 Stage 3 — ADMET & Export"):
            with gr.Row():
                top_admet = gr.Slider(5, 200, value=50, step=1, label="Top N for ADMET")
                use_admetlab_api = gr.Checkbox(label="Use ADMETlab 3.0 API", value=True)
            run_admet_btn = gr.Button("▶ Run ADMET Analysis", variant="primary")
            admet_summary = gr.Textbox(label="Summary", lines=3, interactive=False)
            admet_table = gr.Dataframe(label="ADMET Profiles", wrap=True)
            with gr.Row():
                admet_csv = gr.File(label="⬇ ADMET CSV")
                admet_zip = gr.File(label="⬇ PDB Complexes ZIP")
                admet_docx = gr.File(label="⬇ DOCX Report")

        with gr.Accordion("ℹ️ About DeepDock-AI", open=False):
            gr.Markdown(
                "| Stage | Technology | Output |\n"
                "|---|---|---|\n"
                "| 🔬 ADMET Pre-filter | RDKit · Lipinski RO5 · Veber · PAINS · multiprocessing | Filtered CSV |\n"
                "| ⚗️ Docking | AutoDock-Vina + optional GNINA CNN rescoring | PDB complexes ZIP |\n"
                "| 📊 ADMET Profiling | ADMETlab 3.0 API or RDKit fallback | CSV + DOCX report |\n"
            )

        # ── Wiring ──────────────────────────────────────────────────────────
        run_filter_btn.click(
            fn=run_stage1, inputs=[ligand_input, enable_admet, state],
            outputs=[filter_summary, filter_table, filter_csv, state],
        )
        run_dock_btn.click(
            fn=run_stage2_single,
            inputs=[state, protein_input, conf_input, n_to_dock,
                    exhaustiveness, n_poses, box_size_val, weights_input],
            outputs=[dock_summary, dock_table, dock_zip, state],
        )
        run_batch_btn.click(
            fn=run_stage2_batch,
            inputs=[state, protein_input, conf_input, batch_size, top_k_per_batch,
                    final_exhaustiveness, exhaustiveness, n_poses, box_size_val, weights_input],
            outputs=[batch_summary, batch_table, batch_zip, state],
        )
        run_admet_btn.click(
            fn=run_stage3, inputs=[state, top_admet, use_admetlab_api],
            outputs=[admet_summary, admet_table, admet_csv, admet_zip, admet_docx, state],
        )

    return demo


if __name__ == "__main__":
    gc.collect()
    demo = build_app()
    demo.queue()
    demo.launch()
