"""
DeepDock-AI — GPU-Accelerated Virtual Screening Platform
Main Streamlit application entry point.
"""

from __future__ import annotations
import gc
import io
import time
from typing import List, Optional

import streamlit as st
import pandas as pd
import torch

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DeepDock-AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports ────────────────────────────────────────────────────────────
from modules.preprocessing  import load_ligands_from_bytes, parse_pdb_binding_site, parse_conf_txt, ensure_3d_coords
from modules.filters        import run_parallel_admet_filter
from modules.graph_pipeline import build_dataloader
from modules.model          import load_model, get_device_info
from modules.inference      import run_inference, rank_results
from modules.admet_export   import build_admet_dataframe, export_hits_sdf, generate_docx_report

from rdkit import Chem


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/dna-helix.png",
        width=64,
    )
    st.title("DeepDock-AI")
    st.caption("GPU-Accelerated Virtual Screening")
    st.divider()

    # ── Device info ───────────────────────────────────────────────────────────
    dev_info = get_device_info()
    with st.expander("⚙️ Hardware", expanded=True):
        if dev_info["cuda_available"]:
            st.success(f"🟢 GPU: {dev_info['gpu_name']}")
            st.info(f"VRAM: {dev_info['vram_gb']} GB  |  {dev_info['precision']}")
        else:
            st.warning("🟡 Running on CPU — inference will be slower")
        st.caption(f"PyTorch {torch.__version__}")

    st.divider()

    # ── Screening parameters ──────────────────────────────────────────────────
    st.subheader("🔬 Screening Parameters")

    top_n = st.slider(
        "Top N hits to report", min_value=5, max_value=200, value=50, step=5
    )
    batch_size = st.select_slider(
        "GPU batch size",
        options=[4, 8, 16, 32, 64, 128],
        value=32,
        help="Larger = faster; OOM recovery will halve this automatically.",
    )

    st.divider()

    # ── Molecular property filters ────────────────────────────────────────────
    st.subheader("🧪 Property Filters")

    enable_admet = st.checkbox(
        "Enable 2D ADMET Pre-filter",
        value=True,
        help="Disable for small curated sets (≤20 ligands) to skip Lipinski/PAINS.",
    )

    if enable_admet:
        mw_max   = st.slider("Max MW (Da)",   300, 800,  500, 10)
        logp_max = st.slider("Max LogP",       0.0, 8.0, 5.0, 0.1)
        tpsa_max = st.slider("Max TPSA (Å²)", 60,  200,  140,  5)
        rotb_max = st.slider("Max Rot. Bonds", 2,   20,   10,  1)
    else:
        mw_max = logp_max = tpsa_max = rotb_max = None
        st.info("ADMET filter bypassed — all ligands will be scored.")

    st.divider()

    # ── Model weights (optional) ──────────────────────────────────────────────
    st.subheader("🤖 Model Weights")
    weights_file = st.file_uploader(
        "Upload .pt checkpoint (optional)",
        type=["pt", "pth"],
        help="Leave empty to use the demo GNN with random weights.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═════════════════════════════════════════════════════════════════════════════
st.title("🧬 DeepDock-AI Virtual Screening Platform")
st.markdown(
    "Upload your ligand library and target protein to run GPU-accelerated "
    "binding affinity prediction with real-time ADMET profiling."
)

# ── File upload section ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown("#### 💊 Ligand Library")
    ligand_file = st.file_uploader(
        "Upload Ligands.sdf",
        type=["sdf"],
        help="SDF file with one or more 2D/3D ligand structures.",
        label_visibility="collapsed",
    )
    if ligand_file:
        st.success(f"✓ {ligand_file.name}  ({ligand_file.size / 1024:.1f} KB)")

with col2:
    st.markdown("#### 🎯 Target Protein")
    protein_file = st.file_uploader(
        "Upload protein.pdb",
        type=["pdb"],
        help="PDB file of the target protein. Active-site centroid auto-detected.",
        label_visibility="collapsed",
    )
    if protein_file:
        st.success(f"✓ {protein_file.name}  ({protein_file.size / 1024:.1f} KB)")

with col3:
    st.markdown("#### ⚙️ Grid Config")
    conf_file = st.file_uploader(
        "conf.txt (optional)",
        type=["txt", "conf"],
        help="AutoDock-Vina style config with center_x/y/z and size_x/y/z.",
        label_visibility="collapsed",
    )
    if conf_file:
        st.success(f"✓ {conf_file.name}")

st.divider()

# ── Parsed file previews ──────────────────────────────────────────────────────
if ligand_file or protein_file:
    prev_col1, prev_col2 = st.columns(2)

    if ligand_file:
        with prev_col1:
            with st.expander("📄 Ligand File Preview", expanded=False):
                ligand_bytes = ligand_file.getvalue()
                mols_preview, names_preview = load_ligands_from_bytes(ligand_bytes)
                st.metric("Molecules detected", len(mols_preview))
                if names_preview:
                    st.dataframe(
                        pd.DataFrame({"Name": names_preview[:10]}),
                        height=220,
                        use_container_width=True,
                    )
                    if len(names_preview) > 10:
                        st.caption(f"… and {len(names_preview)-10} more")

    if protein_file:
        with prev_col2:
            with st.expander("🧬 Protein File Preview", expanded=False):
                pdb_bytes = protein_file.getvalue()
                site_info = parse_pdb_binding_site(pdb_bytes)
                cx, cy, cz = site_info["centroid"]
                st.metric("Binding site atoms", site_info["n_atoms"])
                st.metric("Detection method",   site_info["source"])
                st.info(
                    f"**Active-site centroid:** "
                    f"X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f}"
                )

# ── Grid config preview ───────────────────────────────────────────────────────
if conf_file:
    with st.expander("📐 Grid Configuration", expanded=False):
        conf_data = parse_conf_txt(conf_file.getvalue())
        cx, cy, cz = conf_data["center"]
        sx, sy, sz = conf_data["size"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Grid Center:** X={cx}, Y={cy}, Z={cz}")
        with c2:
            st.markdown(f"**Grid Size:** {sx} × {sy} × {sz} Å")


# ═════════════════════════════════════════════════════════════════════════════
# RUN SCREENING
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🚀 Run Screening")

run_disabled = not (ligand_file and protein_file)
run_btn = st.button(
    "▶ Start Virtual Screening",
    type="primary",
    disabled=run_disabled,
    use_container_width=True,
    help="Upload both Ligands.sdf and protein.pdb to enable." if run_disabled else "",
)

if run_disabled:
    st.info("Upload **Ligands.sdf** and **protein.pdb** to begin.")

if run_btn:
    st.divider()
    st.subheader("⚙️ Pipeline Progress")

    # ── Phase 1: Load ligands ─────────────────────────────────────────────────
    with st.status("📥 Loading ligands…", expanded=True) as status_load:
        raw_bytes = ligand_file.getvalue()
        all_mols, all_names = load_ligands_from_bytes(raw_bytes)
        n_input = len(all_mols)
        st.write(f"✅ Loaded **{n_input}** ligands from SDF")

        if n_input == 0:
            st.error("No valid molecules found in SDF. Check file format.")
            st.stop()

        # Warn if small set and filter enabled
        if n_input <= 20 and enable_admet:
            st.warning(
                f"Small set detected ({n_input} ligands). "
                "Consider disabling ADMET pre-filter in the sidebar."
            )

        status_load.update(label="Ligands loaded", state="complete")

    # ── Phase 2: Parse protein + grid ────────────────────────────────────────
    with st.status("🎯 Parsing target protein…", expanded=False) as status_pdb:
        pdb_bytes  = protein_file.getvalue()
        site_info  = parse_pdb_binding_site(pdb_bytes)
        grid_info  = parse_conf_txt(conf_file.getvalue()) if conf_file else None
        cx, cy, cz = site_info["centroid"]
        st.write(
            f"✅ Binding site centroid: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f} "
            f"({site_info['source']}, {site_info['n_atoms']} atoms)"
        )
        status_pdb.update(label="Target parsed", state="complete")

    # ── Phase 3: ADMET pre-filter ─────────────────────────────────────────────
    filter_bar  = st.progress(0.0, text="ADMET filtering…")
    filter_text = st.empty()

    if enable_admet and n_input > 0:
        with st.status("🧪 Running parallel ADMET pre-filter…", expanded=True) as status_filter:
            t0 = time.time()
            passed_mols, admet_records = run_parallel_admet_filter(all_mols)
            elapsed = time.time() - t0
            n_passed = len(passed_mols)
            filter_bar.progress(1.0, text="ADMET filtering complete")
            st.write(
                f"✅ {n_passed}/{n_input} ligands passed  "
                f"(Lipinski + Veber + PAINS)  —  {elapsed:.1f}s"
            )
            status_filter.update(label=f"ADMET filter: {n_passed}/{n_input} passed", state="complete")

        if n_passed == 0:
            st.error(
                "No molecules passed ADMET filters. "
                "Disable the filter in the sidebar or check your ligand set."
            )
            st.stop()

        mols_to_screen = passed_mols
    else:
        filter_bar.progress(1.0, text="ADMET filter skipped")
        filter_text.text("ADMET pre-filter bypassed.")
        mols_to_screen = all_mols
        n_passed = n_input
        admet_records = []

    # ── Phase 4: Load model ───────────────────────────────────────────────────
    with st.status("🤖 Loading GNN model onto GPU…", expanded=False) as status_model:
        weights_path = ""
        if weights_file is not None:
            tmp_path = f"/tmp/deepdock_weights.pt"
            with open(tmp_path, "wb") as f:
                f.write(weights_file.getvalue())
            weights_path = tmp_path

        model, device, precision = load_model(weights_path)
        status_model.update(
            label=f"Model ready — {device} ({precision})",
            state="complete",
        )

    # ── Phase 5: GPU inference ────────────────────────────────────────────────
    st.markdown("#### 🔬 GPU Inference")
    progress_bar  = st.progress(0.0, text="Scoring ligands…")
    status_text   = st.empty()

    with st.spinner("Running GNN docking inference…"):
        t0 = time.time()
        scored_mols, scores = run_inference(
            model        = model,
            device       = device,
            mols         = mols_to_screen,
            batch_size   = batch_size,
            use_fp16     = (device.type == "cuda"),
            progress_bar = progress_bar,
            status_text  = status_text,
        )
        elapsed_infer = time.time() - t0

    progress_bar.progress(1.0, text="Inference complete")
    status_text.success(
        f"✅ Scored {len(scored_mols)} ligands in {elapsed_infer:.1f}s  "
        f"({len(scored_mols)/max(elapsed_infer,0.01):.0f} ligands/sec)"
    )

    # ── Phase 6: Rank & ADMET profiling ──────────────────────────────────────
    with st.status("📊 Ranking hits & computing ADMET profiles…", expanded=False) as status_rank:
        top_mols, top_scores = rank_results(scored_mols, scores, top_n=top_n)
        top_names = [
            m.GetPropsAsDict().get("_Name", f"hit_{i+1}")
            for i, m in enumerate(top_mols)
        ]
        admet_df = build_admet_dataframe(top_mols, top_scores, top_names)
        status_rank.update(label=f"Top {len(top_mols)} hits ranked", state="complete")

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🏆 Screening Results")

    # KPI metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Input ligands",    n_input)
    m2.metric("Passed ADMET",     n_passed)
    m3.metric("Scored",           len(scored_mols))
    m4.metric("Best ΔG (kcal/mol)",
              f"{min(top_scores):.3f}" if top_scores else "N/A")
    m5.metric("Mean QED",
              f"{admet_df['QED'].mean():.3f}" if "QED" in admet_df else "N/A")

    st.divider()

    # ── Interactive results table ─────────────────────────────────────────────
    st.markdown("#### 📋 Top Hit Compounds — Interactive Table")

    # Colour map helper
    def _style_dg(val):
        if isinstance(val, (int, float)):
            if val < -10:   return "color: #00C853; font-weight:bold"
            elif val < -7:  return "color: #64DD17"
            elif val < -5:  return "color: #FFD600"
            else:           return "color: #FF6D00"
        return ""

    styled = admet_df.style.applymap(
        _style_dg, subset=["ΔG (kcal/mol)"]
    ).format({
        "ΔG (kcal/mol)": "{:.3f}",
        "MW":  "{:.1f}",
        "LogP":"{:.2f}",
        "QED": "{:.3f}",
        "LogS":"{:.2f}",
        "TPSA":"{:.1f}",
    }, na_rep="N/A")

    st.dataframe(
        styled,
        use_container_width=True,
        height=420,
    )

    # ── Score distribution ────────────────────────────────────────────────────
    with st.expander("📈 Score Distribution", expanded=True):
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**ΔG Distribution**")
            st.bar_chart(
                pd.DataFrame({"ΔG (kcal/mol)": top_scores}),
                y="ΔG (kcal/mol)",
                use_container_width=True,
            )
        with dist_col2:
            st.markdown("**QED Distribution**")
            if "QED" in admet_df.columns:
                st.bar_chart(
                    admet_df[["QED"]].reset_index(drop=True),
                    use_container_width=True,
                )

    # ── ADMET detail table ────────────────────────────────────────────────────
    with st.expander("🧪 Full ADMET Profiles", expanded=False):
        st.dataframe(admet_df, use_container_width=True, height=380)

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT SECTION
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📦 Export Results")

    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

    # CSV
    with exp_col1:
        csv_bytes = admet_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name="deepdock_hits.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Parquet
    with exp_col2:
        parquet_buf = io.BytesIO()
        admet_df.to_parquet(parquet_buf, index=False, engine="pyarrow")
        st.download_button(
            "⬇ Download Parquet",
            data=parquet_buf.getvalue(),
            file_name="deepdock_hits.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )

    # SDF
    with exp_col3:
        with st.spinner("Building SDF…"):
            sdf_bytes = export_hits_sdf(
                top_mols, top_scores, top_names, admet_df
            )
        st.download_button(
            "⬇ Download SDF (3D)",
            data=sdf_bytes,
            file_name="top_hits_docked.sdf",
            mime="chemical/x-mdl-sdfile",
            use_container_width=True,
        )

    # DOCX
    with exp_col4:
        with st.spinner("Generating DOCX report…"):
            try:
                docx_bytes = generate_docx_report(
                    admet_df        = admet_df,
                    n_input         = n_input,
                    n_after_filter  = n_passed,
                    top_n           = len(top_mols),
                    protein_info    = site_info,
                    grid_info       = grid_info,
                )
                st.download_button(
                    "⬇ Download DOCX Report",
                    data=docx_bytes,
                    file_name="Screening_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"DOCX generation failed: {e}")

    st.success("✅ Screening complete! Download your results above.")

    # ── Memory cleanup ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# ABOUT / FOOTER
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ About DeepDock-AI", expanded=False):
    st.markdown("""
    **DeepDock-AI** is a GPU-accelerated virtual screening platform combining:

    | Component | Technology |
    |-----------|-----------|
    | UI        | Streamlit |
    | Docking GNN | PyTorch (GCN + Attention Pooling) |
    | GPU Accel | CUDA FP16 AMP (Kaggle T4 / P100) |
    | Cheminformatics | RDKit |
    | Pre-filter | Lipinski RO5 + Veber + PAINS (multiprocessing) |
    | ADMET | QED, TPSA, LogP, MW, ESOL LogS |
    | Export | CSV, Parquet, SDF (3D), DOCX report |

    **Workflow:** Upload SDF → ADMET filter → Graph encoding → GNN scoring → Rank & export

    **Batch OOM recovery:** If CUDA runs out of memory, batch size is automatically halved and inference retries.
    """)

st.caption("DeepDock-AI  |  PyTorch + RDKit  |  Built for Kaggle T4/P100 GPUs")
