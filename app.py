"""
DeepDock-AI — GPU-Accelerated Virtual Screening Platform
3-Stage Pipeline: RDKit Filter → GNINA Docking → ADMETlab 3.0
FIXED VERSION — All Tab 2 bugs resolved
"""

from __future__ import annotations
import gc
import io
import os
import tempfile
import time
from typing import List, Optional, Tuple
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
from modules.preprocessing import (
    load_ligands_from_bytes,
    parse_pdb_binding_site,
    parse_conf_txt,
    ensure_3d_coords,
)
from modules.filters import run_parallel_admet_filter, compute_single_admet
from modules.docking import dock_ligands, load_gnina_model, VINA_AVAILABLE
from modules.admetlab import run_admet_analysis
from modules.export import (
    export_pdb_complexes_zip,
    export_csv,
    generate_docx_report,
)
from modules.model import get_device_info
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═════════════════════════════════════════════════════════════════════════════
for key in [
    "filtered_mols",
    "filter_records",
    "docking_results",
    "admet_df",
    "admet_source",
    "protein_info",
    "grid_info",
    "all_mols",
    "all_names",
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 DeepDock-AI")
    st.caption("GPU-Accelerated Virtual Screening")
    st.divider()

    # ── Hardware status ───────────────────────────────────────────────────────
    dev = get_device_info()
    with st.expander("⚙️ Hardware", expanded=True):
        if dev["cuda_available"]:
            st.success(f"🟢 {dev['gpu_name']}")
            st.info(f"VRAM {dev['vram_gb']} GB · {dev['precision']}")
        else:
            st.warning("🟡 CPU mode — docking will be slower")
        st.caption(f"PyTorch {torch.__version__}")
        if VINA_AVAILABLE:
            st.success("✅ AutoDock-Vina ready")
        else:
            st.error("❌ vina not found — CNN-only mode")

    st.divider()

    # ── Stage 1 params ────────────────────────────────────────────────────────
    st.markdown("**Stage 1 — Filter**")
    enable_admet = st.checkbox(
        "Enable ADMET pre-filter",
        value=True,
        help="Disable for curated sets ≤20 ligands",
    )

    st.divider()

    # ── Stage 2 params ────────────────────────────────────────────────────────
    st.markdown("**Stage 2 — Docking**")
    exhaustiveness = st.slider("Vina exhaustiveness", 1, 32, 8)
    n_poses = st.slider("Poses per ligand", 1, 20, 5)
    batch_dock = st.slider("Dock top N (from filter)", 10, 500, 100)
    box_size_val = st.slider("Grid box size (Å)", 10, 40, 20)

    st.divider()

    # ── Stage 3 params ────────────────────────────────────────────────────────
    st.markdown("**Stage 3 — ADMET**")
    use_admetlab_api = st.checkbox(
        "Use ADMETlab 3.0 API",
        value=True,
        help="Falls back to RDKit if API unreachable",
    )
    top_admet = st.slider("Top N for ADMET", 5, 200, 50)

    st.divider()

    # ── GNINA weights (optional) ──────────────────────────────────────────────
    st.markdown("**GNINA Weights** *(optional)*")
    weights_file = st.file_uploader(
        "Upload .pt checkpoint", type=["pt", "pth"], label_visibility="collapsed"
    )
    st.caption("Leave empty → random-init CNN (Vina score used)")


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("🧬 DeepDock-AI")
st.markdown(
    "**3-Stage Pipeline:**  "
    "🔬 RDKit ADMET Filter  →  ⚗️ GNINA Molecular Docking  →  📊 ADMETlab 3.0 Profiling"
)

# ── File uploaders ────────────────────────────────────────────────────────────
uc1, uc2, uc3 = st.columns([2, 2, 1])
with uc1:
    st.markdown("#### 💊 Ligand Library")
    ligand_file = st.file_uploader(
        "Ligands.sdf", type=["sdf"], label_visibility="collapsed"
    )
    if ligand_file:
        st.success(f"✓ {ligand_file.name}  ({ligand_file.size / 1024:.1f} KB)")

with uc2:
    st.markdown("#### 🎯 Target Protein")
    protein_file = st.file_uploader(
        "protein.pdbqt", type=["pdbqt"], label_visibility="collapsed"
    )
    if protein_file:
        st.success(f"✓ {protein_file.name}  ({protein_file.size / 1024:.1f} KB)")

with uc3:
    st.markdown("#### ⚙️ Grid Config")
    conf_file = st.file_uploader(
        "conf.txt *(optional)*", type=["txt", "conf"], label_visibility="collapsed"
    )
    if conf_file:
        st.success(f"✓ {conf_file.name}")

# Quick previews
if ligand_file or protein_file:
    pv1, pv2 = st.columns(2)
    if ligand_file:
        with pv1:
            with st.expander("📄 Ligand preview"):
                preview_mols, preview_names = load_ligands_from_bytes(
                    ligand_file.getvalue()
                )
                st.metric("Molecules detected", len(preview_mols))
                if preview_names:
                    st.dataframe(
                        pd.DataFrame({"Name": preview_names[:8]}),
                        height=200,
                        use_container_width=True,
                    )
    if protein_file:
        with pv2:
            with st.expander("🧬 Protein preview"):
                site = parse_pdb_binding_site(protein_file.getvalue())
                cx, cy, cz = site["centroid"]
                st.metric("Binding-site atoms", site["n_atoms"])
                st.info(
                    f"Centroid: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f} ({site['source']})"
                )

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  3-STAGE PIPELINE TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(
    [
        "🔬 Stage 1 — RDKit Filter",
        "⚗️ Stage 2 — GNINA Docking",
        "📊 Stage 3 — ADMET & Export",
    ]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — RDKit ADMET Pre-Filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("🔬 Stage 1 — Parallel ADMET Pre-Filtering")
    st.markdown(
        "Filters compounds using **Lipinski Rule of 5**, **Veber oral bioavailability "
        "rules**, and **PAINS A/B/C catalog** via RDKit — executed in parallel with "
        "`multiprocessing.Pool`."
    )

    if not ligand_file:
        st.info("Upload **Ligands.sdf** to begin filtering.")
    else:
        run_filter = st.button(
            "▶ Run RDKit Filter", type="primary", use_container_width=True
        )
        if run_filter:
            with st.status("Loading ligands…", expanded=True) as s1:
                raw_mols, raw_names = load_ligands_from_bytes(ligand_file.getvalue())
                n_in = len(raw_mols)
                st.write(f"✅ Loaded **{n_in}** ligands")
                if n_in == 0:
                    s1.update(label="No molecules found", state="error")
                    st.error("No valid molecules found.")
                    st.stop()
                if n_in <= 20 and enable_admet:
                    st.warning(
                        f"Small set ({n_in} ligands) — consider disabling filter."
                    )
                s1.update(label="Ligands loaded", state="complete")

            if enable_admet:
                with st.status("Running parallel ADMET filter…", expanded=True) as s2:
                    t0 = time.time()
                    passed, records = run_parallel_admet_filter(raw_mols)
                    elapsed = time.time() - t0
                    st.write(
                        f"✅ **{len(passed)}/{n_in}** passed  "
                        f"(Lipinski + Veber + PAINS) — {elapsed:.1f}s"
                    )
                    s2.update(
                        label=f"Filter complete: {len(passed)}/{n_in}", state="complete"
                    )
            else:
                passed, records = raw_mols, []
                st.info("ADMET filter bypassed — all ligands passed.")

            st.session_state.filtered_mols = passed
            st.session_state.filter_records = records
            st.session_state.all_mols = raw_mols
            st.session_state.all_names = raw_names

        # ── Show results ──────────────────────────────────────────────────────
        if st.session_state.filtered_mols is not None:
            n_pass = len(st.session_state.filtered_mols)
            n_tot = len(st.session_state.all_mols) if st.session_state.all_mols else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Input ligands", n_tot)
            m2.metric("Passed filter", n_pass)
            m3.metric("Rejected", n_tot - n_pass)
            m4.metric("Pass rate", f"{n_pass / max(n_tot, 1) * 100:.1f}%")

            if st.session_state.filter_records:
                st.markdown("#### Filter Report")
                df_filt = pd.DataFrame(st.session_state.filter_records)

                def _colour_pass(val):
                    if val is True:
                        return "color:#00C853;font-weight:bold"
                    if val is False:
                        return "color:#FF1744"
                    return ""

                bool_cols = [
                    c
                    for c in df_filt.columns
                    if c in ("LipinskiPass", "VeberPass", "PAINSPass", "AllPass")
                ]
                styled_f = df_filt.style.map(_colour_pass, subset=bool_cols)
                st.dataframe(styled_f, use_container_width=True, height=380)

                csv_filt = export_csv(df_filt)
                st.download_button(
                    "⬇ Download Filter Report CSV",
                    csv_filt,
                    "filter_report.csv",
                    "text/csv",
                )

            st.success(
                f"✅ {n_pass} ligands ready for docking. Proceed to **Stage 2**."
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — GNINA Docking (FIXED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("⚗️ Stage 2 — GNINA Molecular Docking")
    st.markdown(
        "**AutoDock-Vina** sampling + **GNINA CNN** (Ragoza et al. 2017) rescoring.  "
        "OOM-safe FP16 AMP inference on CUDA. Results exported as PDB complexes."
    )

    if not VINA_AVAILABLE:
        st.warning(
            "⚠️ AutoDock-Vina Python package not installed.  \n"
            "Running in **GNINA CNN-only** scoring mode (no 3D pose generation).  \n"
            "To enable full docking: `pip install vina meeko`"
        )

    mols_ready = st.session_state.filtered_mols
    if mols_ready is None:
        st.info("Complete **Stage 1** first, or upload files below.")
    elif len(mols_ready) == 0:
        st.error("No ligands passed filtering. Go back to Stage 1.")
    elif not protein_file:
        st.info("Upload **protein.pdbqt** to run docking.")
    else:
        n_to_dock = min(batch_dock, len(mols_ready))
        dock_mols = mols_ready[:n_to_dock]
        dock_names = (
            st.session_state.all_names or [f"mol_{i}" for i in range(n_to_dock)]
        )[:n_to_dock]

        st.info(
            f"Ready to dock **{n_to_dock}** ligands "
            f"(top {batch_dock} from filter)  ·  "
            f"exhaustiveness={exhaustiveness}  ·  {n_poses} poses/ligand"
        )

        run_dock = st.button(
            "▶ Run GNINA Docking", type="primary", use_container_width=True
        )
        if run_dock:
            pdbqt_bytes = protein_file.getvalue()
            site_info = parse_pdb_binding_site(pdbqt_bytes)
            grid_info = parse_conf_txt(conf_file.getvalue()) if conf_file else None

            # Determine grid center
            if grid_info:
                center = grid_info["center"]
                box_size = grid_info["size"]
            else:
                center = site_info["centroid"]
                box_size = (box_size_val, box_size_val, box_size_val)

            cx, cy, cz = center
            st.info(
                f"Grid center: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f}  |  "
                f"Box: {box_size[0]}×{box_size[1]}×{box_size[2]} Å"
            )

            weights_path = ""
            if weights_file:
                tmp = os.path.join(tempfile.gettempdir(), "gnina_weights.pt")
                with open(tmp, "wb") as f:
                    f.write(weights_file.getvalue())
                weights_path = tmp

            prog_dock = st.progress(0.0, text="Starting docking…")
            stat_dock = st.empty()

            # ✅ FIX: Added proper error handling
            with st.spinner("Running GNINA docking…"):
                t0 = time.time()

                try:
                    results = dock_ligands(
                        mols=dock_mols,
                        names=dock_names,
                        pdb_bytes=pdbqt_bytes,  # ✓ PDBQT bytes directly
                        center=center,
                        box_size=box_size,
                        exhaustiveness=exhaustiveness,
                        n_poses=n_poses,
                        progress_bar=prog_dock,
                        status_text=stat_dock,
                        gnina_weights=weights_path,
                    )
                    elapsed_dock = time.time() - t0

                    prog_dock.progress(1.0)
                    stat_dock.success(
                        f"✅ Docked {len(results)} ligands in {elapsed_dock:.1f}s  "
                        f"({len(results) / max(elapsed_dock, 0.01):.1f} lig/sec)"
                    )

                    st.session_state.docking_results = results
                    st.session_state.protein_info = site_info
                    st.session_state.grid_info = grid_info

                except Exception as e:
                    st.error(f"❌ Docking failed: {str(e)}")
                    stat_dock.error(f"Error: {str(e)}")
                    import traceback

                    st.write("```")
                    st.write(traceback.format_exc())
                    st.write("```")

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        # ── Show docking results ──────────────────────────────────────────────
        if st.session_state.docking_results is not None:
            results = st.session_state.docking_results

            if len(results) == 0:
                st.warning(
                    "⚠️ Docking ran but no ligands produced valid results. Check grid center/box size or protein file."
                )
            else:
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Ligands docked", len(results))
                if results:
                    best = min(r.gnina_affinity for r in results)
                    d2.metric("Best ΔG (kcal/mol)", f"{best:.3f}")
                    mean = sum(r.gnina_affinity for r in results) / len(results)
                    d3.metric("Mean ΔG (kcal/mol)", f"{mean:.3f}")
                    has_poses = sum(1 for r in results if r.pose_pdb)
                    d4.metric("3D poses generated", has_poses)

                st.markdown("#### Docking Results Table")
                dock_records = [
                    {
                        "Rank": i + 1,
                        "Name": r.name,
                        "Vina Score": round(r.vina_score, 3),
                        "GNINA ΔG (kcal/mol)": round(r.gnina_affinity, 3),
                        "3D Pose": "✓" if r.pose_pdb else "–",
                    }
                    for i, r in enumerate(results)
                ]

                dock_df = pd.DataFrame(dock_records)

                # ✅ FIX: Proper styling without broken applymap
                def _dg_colour(val):
                    if not isinstance(val, (int, float)):
                        return ""
                    if val < -10:
                        return "color:#00C853;font-weight:bold"
                    if val < -7:
                        return "color:#64DD17"
                    if val < -5:
                        return "color:#FFD600"
                    return "color:#FF6D00"

                styled_d = dock_df.style.map(
                    lambda val: _dg_colour(val),
                    subset=["GNINA ΔG (kcal/mol)", "Vina Score"],
                )

                st.dataframe(styled_d, use_container_width=True, height=420)

                # ── PDB complex download ──────────────────────────────────────────
                complex_pdbs = [r.complex_pdb for r in results if r.complex_pdb]
                names_dock = [r.name for r in results if r.complex_pdb]
                scores_dock = [r.gnina_affinity for r in results if r.complex_pdb]

                if complex_pdbs:
                    zip_bytes = export_pdb_complexes_zip(
                        complex_pdbs, names_dock, scores_dock
                    )
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        st.download_button(
                            "⬇ Download PDB Complexes (ZIP)",
                            zip_bytes,
                            "docked_complexes.zip",
                            "application/zip",
                            use_container_width=True,
                        )
                    with ec2:
                        st.download_button(
                            "⬇ Download Docking CSV",
                            export_csv(dock_df),
                            "docking_results.csv",
                            "text/csv",
                            use_container_width=True,
                        )

                st.success(
                    "✅ Docking complete. Proceed to **Stage 3** for ADMET analysis."
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — ADMET & Export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("📊 Stage 3 — ADMET Analysis & Export")
    st.markdown(
        "ADMET profiling via **ADMETlab 3.0** (Gui et al. 2024) API with RDKit fallback.  "
        "Exports: **DOCX report**, **CSV**, **PDB complexes ZIP**."
    )

    docking_results = st.session_state.docking_results
    if docking_results is None:
        st.info("Complete **Stage 2** first to generate docking results.")
    elif len(docking_results) == 0:
        st.warning("⚠️ No ligands were successfully docked in Stage 2. Go back and check your grid, box size, or protein file.")
    else:
        top_results = docking_results[:top_admet]
        top_mols = [r.mol for r in top_results]
        top_names = [r.name for r in top_results]
        top_scores = [r.gnina_affinity for r in top_results]

        st.info(
            f"Analysing top **{len(top_results)}** hits by GNINA ΔG.  "
            f"ADMET source: {'ADMETlab 3.0 API' if use_admetlab_api else 'RDKit'}"
        )

        run_admet = st.button(
            "▶ Run ADMET Analysis", type="primary", use_container_width=True
        )
        if run_admet:
            stat_admet = st.empty()
            with st.spinner("Running ADMET analysis…"):
                t0 = time.time()
                admet_df, source = run_admet_analysis(
                    mols=top_mols,
                    names=top_names,
                    scores=top_scores,
                    use_api=use_admetlab_api,
                    status_text=stat_admet,
                )
                elapsed_a = time.time() - t0

            stat_admet.success(
                f"✅ ADMET complete — source: **{source}** — {elapsed_a:.1f}s"
            )
            st.session_state.admet_df = admet_df
            st.session_state.admet_source = source

        # ── Show ADMET results ────────────────────────────────────────────────
        if st.session_state.admet_df is not None:
            admet_df = st.session_state.admet_df
            source = st.session_state.admet_source or "RDKit"

            # KPI row
            k1, k2, k3, k4, k5 = st.columns(5)
            dg_col = admet_df.get("ΔG (kcal/mol)", pd.Series(dtype=float))
            k1.metric("Compounds analysed", len(admet_df))
            k2.metric(
                "Best ΔG (kcal/mol)", f"{dg_col.min():.3f}" if not dg_col.empty else "—"
            )
            if "QED" in admet_df:
                k3.metric("Median QED", f"{admet_df['QED'].median():.3f}")
            if "DrugLikeness" in admet_df:
                excellent = (admet_df["DrugLikeness"] == "Excellent").sum()
                k4.metric("Excellent drug-like", excellent)
            if "BBB" in admet_df:
                bbb_yes = (admet_df["BBB"] == "Yes").sum()
                k5.metric("BBB-penetrant", bbb_yes)

            st.markdown(f"#### ADMET Profiles *(source: {source})*")

            # Colour helpers
            def _dl_colour(val):
                mapping = {
                    "Excellent": "#00C853",
                    "Good": "#64DD17",
                    "Moderate": "#FFD600",
                    "Poor": "#FF6D00",
                }
                c = mapping.get(str(val), "")
                return f"color:{c};font-weight:bold" if c else ""

            def _pass_colour(val):
                s = str(val)
                if s in ("Pass", "Yes", "Low Risk", "Clean"):
                    return "color:#00C853"
                if s in ("Fail", "No", "High Risk", "Alert"):
                    return "color:#FF1744"
                if s == "Medium Risk":
                    return "color:#FFD600"
                return ""

            style_cols = [
                c
                for c in admet_df.columns
                if any(
                    k in c
                    for k in (
                        "Pass",
                        "Fail",
                        "BBB",
                        "Risk",
                        "Clean",
                        "Alert",
                        "Penetration",
                        "Like",
                    )
                )
            ]
            st_admet = admet_df.style
            if "DrugLikeness" in admet_df.columns:
                st_admet = st_admet.map(_dl_colour, subset=["DrugLikeness"])
            if style_cols:
                st_admet = st_admet.map(_pass_colour, subset=style_cols)

            st.dataframe(st_admet, use_container_width=True, height=420)

            # Drug-likeness distribution
            if "DrugLikeness" in admet_df.columns:
                with st.expander("📈 Drug-Likeness Distribution"):
                    dl_counts = admet_df["DrugLikeness"].value_counts()
                    st.bar_chart(dl_counts, use_container_width=True)

            st.divider()

            # ── Export buttons ────────────────────────────────────────────────
            st.markdown("#### 📦 Export Results")
            e1, e2, e3 = st.columns(3)

            # CSV
            with e1:
                st.download_button(
                    "⬇ Download ADMET CSV",
                    export_csv(admet_df),
                    "admet_results.csv",
                    "text/csv",
                    use_container_width=True,
                )

            # PDB complexes ZIP
            dock_res = st.session_state.docking_results or []
            top_dock = dock_res[:top_admet]
            complexes = [r.complex_pdb for r in top_dock if r.complex_pdb]
            c_names = [r.name for r in top_dock if r.complex_pdb]
            c_scores = [r.gnina_affinity for r in top_dock if r.complex_pdb]
            if complexes:
                with e2:
                    pdb_zip = export_pdb_complexes_zip(complexes, c_names, c_scores)
                    st.download_button(
                        "⬇ PDB Complexes ZIP",
                        pdb_zip,
                        "top_hits_complexes.zip",
                        "application/zip",
                        use_container_width=True,
                    )

            # DOCX report
            with e3:
                with st.spinner("Generating DOCX report…"):
                    try:
                        # Build docking summary df
                        dock_records = [
                            {
                                "Rank": i + 1,
                                "Name": r.name,
                                "Vina Score": round(r.vina_score, 3),
                                "GNINA ΔG (kcal/mol)": round(r.gnina_affinity, 3),
                            }
                            for i, r in enumerate(top_dock)
                        ]
                        dock_summary_df = pd.DataFrame(dock_records)

                        filter_df_out = (
                            pd.DataFrame(st.session_state.filter_records)
                            if st.session_state.filter_records
                            else None
                        )

                        docx_bytes = generate_docx_report(
                            filter_df=filter_df_out,
                            docking_df=dock_summary_df,
                            admet_df=admet_df,
                            admet_source=source,
                            n_input=(
                                len(st.session_state.all_mols)
                                if st.session_state.all_mols
                                else 0
                            ),
                            n_filtered=(
                                len(st.session_state.filtered_mols)
                                if st.session_state.filtered_mols
                                else 0
                            ),
                            n_docked=len(dock_res),
                            protein_info=st.session_state.protein_info,
                            grid_info=st.session_state.grid_info,
                            top_n=top_admet,
                        )
                        st.download_button(
                            "⬇ Download DOCX Report",
                            docx_bytes,
                            "DeepDock_AI_Report.docx",
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document",
                            use_container_width=True,
                        )
                    except Exception as ex:
                        st.error(f"DOCX error: {ex}")

            st.success("✅ All results ready for download!")


# ═════════════════════════════════════════════════════════════════════════════
#  ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ About DeepDock-AI", expanded=False):
    st.markdown("""
| Stage | Technology | Output |
|-------|-----------|--------|
| 🔬 ADMET Pre-filter | RDKit · Lipinski RO5 · Veber · PAINS · multiprocessing | Filtered SDF + CSV |
| ⚗️ Docking | AutoDock-Vina sampling + GNINA CNN rescoring (PyTorch FP16 AMP) | PDB complexes ZIP |
| 📊 ADMET Profiling | ADMETlab 3.0 API (Gui et al. 2024) or RDKit fallback | CSV + DOCX report |

**GNINA CNN** (Ragoza et al. *J. Chem. Inf. Model.* 2017):  
3D CNN (35-ch voxel grid, 24³) → Pose quality + Binding affinity (ΔG kcal/mol)  
Weights: [github.com/gnina/gnina/tree/master/weights](https://github.com/gnina/gnina/tree/master/weights)

**Full GNINA binary** on Kaggle: `conda install -c conda-forge gnina`

**Optimised for:** Kaggle Dual T4 / P100 · CUDA FP16 AMP · CUDA OOM auto-recovery
    """)

st.caption(
    "DeepDock-AI · AutoDock-Vina + GNINA + RDKit + ADMETlab 3.0 · Kaggle T4/P100"
)
