"""
Export Module: DOCX report, CSV, and PDB complex files (ZIP archive).
FIXED VERSION — Complete with pdbqt_to_pdb_standard function
"""

from __future__ import annotations
import io
import math
import zipfile
import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

try:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    DOCX_OK = True
except ImportError:
    DOCX_OK = False


# ── PDBQT to PDB Conversion ───────────────────────────────────────────────────
def pdbqt_to_pdb_standard(pdbqt_content: str) -> str:
    """
    Convert PDBQT format (with partial charges) to standard PDB format.
    Keeps only ATOM/HETATM lines and strips PDBQT-specific columns.
    """
    pdb_lines = []
    for line in pdbqt_content.splitlines():
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            # Keep first 66 characters (standard PDB format)
            pdb_lines.append(line[:66])
    return "\n".join(pdb_lines) + "\n"


# ── PDB Complex ZIP ───────────────────────────────────────────────────────────
def export_pdb_complexes_zip(
    complex_pdbs: List[str],
    names: List[str],
    scores: List[float],
) -> bytes:
    """
    Pack all protein-ligand PDB complexes into a single ZIP file.
    Each file: <rank>_<name>_complex.pdb with REMARK headers.
    Returns raw ZIP bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rank, (pdb_str, name, score) in enumerate(
            zip(complex_pdbs, names, scores), start=1
        ):
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            fname = f"{rank:03d}_{safe_name}_complex.pdb"

            header = (
                f"REMARK   DeepDock-AI Virtual Screening Result\n"
                f"REMARK   Rank       : {rank}\n"
                f"REMARK   Compound   : {name}\n"
                f"REMARK   GNINA ΔG   : {score:.3f} kcal/mol\n"
                f"REMARK   Date       : {datetime.datetime.utcnow().strftime('%Y-%m-%d')}\n"
                f"REMARK   Software   : AutoDock-Vina + GNINA CNN Rescoring\n"
                f"REMARK\n"
            )
            zf.writestr(fname, header + (pdb_str or ""))
    buf.seek(0)
    return buf.getvalue()


# ── CSV Export ────────────────────────────────────────────────────────────────
def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ── DOCX Report ───────────────────────────────────────────────────────────────
def _cell_bg(cell, hex_color: str):
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    shading = parse_xml(
        f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{hex_color}"/>'
    )
    cell._tc.get_or_add_tcPr().append(shading)


def _add_section_heading(doc, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    if h.runs:
        h.runs[0].font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)
    return h


def generate_docx_report(
    filter_df: Optional[pd.DataFrame],
    docking_df: pd.DataFrame,
    admet_df: pd.DataFrame,
    admet_source: str,
    n_input: int,
    n_filtered: int,
    n_docked: int,
    protein_info: Optional[Dict] = None,
    grid_info: Optional[Dict] = None,
    top_n: int = 50,
) -> bytes:
    """Generate a comprehensive DOCX screening report across all 3 pipeline stages."""
    if not DOCX_OK:
        raise ImportError("python-docx not installed")

    doc = Document()
    sec = doc.sections[0]
    for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, attr, Inches(0.9))

    # ── Cover ─────────────────────────────────────────────────────────────────
    t = doc.add_heading("DeepDock-AI Virtual Screening Report", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if t.runs:
        t.runs[0].font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)
        t.runs[0].font.size = Pt(22)

    sub = doc.add_paragraph(
        f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Pipeline: RDKit Filter → GNINA Docking → {admet_source} ADMET"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        sub.runs[0].font.size = Pt(10)
    doc.add_paragraph()

    # ── 1. Pipeline Summary ───────────────────────────────────────────────────
    _add_section_heading(doc, "1. Pipeline Summary")

    # ✅ FIX: Use proper DataFrame column access
    if "GNINA ΔG (kcal/mol)" in docking_df.columns:
        dg_col = docking_df["GNINA ΔG (kcal/mol)"]
    elif "ΔG (kcal/mol)" in docking_df.columns:
        dg_col = docking_df["ΔG (kcal/mol)"]
    else:
        dg_col = pd.Series(dtype=float)

    best_dg = f"{dg_col.min():.3f}" if not dg_col.empty else "N/A"
    mean_dg = f"{dg_col.mean():.3f}" if not dg_col.empty else "N/A"
    pass_rate = f"{n_filtered / max(n_input, 1) * 100:.1f}%"

    summary_rows = [
        ("Total ligands uploaded", n_input),
        ("Passed ADMET pre-filter", n_filtered),
        ("Filter pass rate", pass_rate),
        ("Ligands docked", n_docked),
        ("Top hits analysed (ADMET)", len(admet_df)),
        ("Best GNINA ΔG (kcal/mol)", best_dg),
        ("Mean GNINA ΔG (kcal/mol)", mean_dg),
        ("ADMET source", admet_source),
    ]
    _table_two_col(doc, summary_rows, header_color="0D47A1")
    doc.add_paragraph()

    # ── 2. Target & Grid ─────────────────────────────────────────────────────
    if protein_info or grid_info:
        _add_section_heading(doc, "2. Target Protein & Docking Grid")
        if protein_info:
            cx, cy, cz = protein_info.get("centroid", (0, 0, 0))
            doc.add_paragraph(
                f"Binding-site centroid (auto-detected): "
                f"X={cx:.2f} Å, Y={cy:.2f} Å, Z={cz:.2f} Å  "
                f"[{protein_info.get('source', '?')}, "
                f"{protein_info.get('n_atoms', 0)} atoms]"
            )
        if grid_info:
            cx, cy, cz = grid_info.get("center", (0, 0, 0))
            sx, sy, sz = grid_info.get("size", (20, 20, 20))
            doc.add_paragraph(
                f"Docking grid center: X={cx}, Y={cy}, Z={cz}  |  "
                f"Box size: {sx}×{sy}×{sz} Å"
            )
        doc.add_paragraph()

    # ── 3. RDKit Filter Results ───────────────────────────────────────────────
    if filter_df is not None and not filter_df.empty:
        _add_section_heading(doc, "3. RDKit ADMET Pre-Filter Results")
        doc.add_paragraph(
            f"{n_filtered} of {n_input} compounds passed "
            f"Lipinski RO5 + Veber + PAINS filters."
        )
        _df_to_table(
            doc,
            filter_df.head(30),
            max_rows=30,
            even_color="E3F2FD",
            header_color="1565C0",
        )
        doc.add_paragraph()

    # ── 4. Docking Results ────────────────────────────────────────────────────
    _add_section_heading(doc, "4. GNINA Docking Results (Top Hits)")
    doc.add_paragraph(
        f"Compounds sorted by GNINA affinity (most negative = strongest binding). "
        f"Scoring: AutoDock-Vina sampling + GNINA CNN rescoring."
    )
    _df_to_table(
        doc,
        docking_df.head(top_n),
        max_rows=top_n,
        even_color="E8F5E9",
        header_color="2E7D32",
    )
    doc.add_paragraph()

    # ── 5. ADMET Profiles ────────────────────────────────────────────────────
    _add_section_heading(doc, f"5. ADMET Profiles ({admet_source})")
    _df_to_table(
        doc,
        admet_df.head(top_n),
        max_rows=top_n,
        even_color="FFF3E0",
        header_color="E65100",
        font_size=7,
    )
    doc.add_paragraph()

    # ── 6. Drug-likeness summary ──────────────────────────────────────────────
    if "DrugLikeness" in admet_df.columns:
        _add_section_heading(doc, "6. Drug-Likeness Distribution")
        for label, count in admet_df["DrugLikeness"].value_counts().items():
            doc.add_paragraph(
                f"  {label}: {count} compounds ({count / len(admet_df) * 100:.1f}%)",
                style="List Bullet",
            )
        doc.add_paragraph()

    # ── 7. Methods ────────────────────────────────────────────────────────────
    _add_section_heading(doc, "7. Methods")
    p = doc.add_paragraph()
    _bold_run(p, "Step 1 — RDKit Pre-filter: ")
    p.add_run(
        "Lipinski Rule of Five (MW≤500, LogP≤5, HBD≤5, HBA≤10), "
        "Veber rules (RotBonds≤10, TPSA≤140), and PAINS A/B/C catalog. "
        "Parallel execution via Python multiprocessing.\n"
    )
    _bold_run(p, "Step 2 — GNINA Docking: ")
    p.add_run(
        "AutoDock-Vina for pose sampling (exhaustiveness=8). "
        "GNINA CNN (Ragoza et al. 2017) for affinity rescoring. "
        "CUDA FP16 AMP acceleration where available.\n"
    )
    _bold_run(p, "Step 3 — ADMET Analysis: ")
    p.add_run(
        f"{'ADMETlab 3.0 API (Gui et al. Nucleic Acids Res. 2024)' if admet_source == 'ADMETlab 3.0' else 'RDKit descriptor calculations (ADMETlab 3.0 API unavailable)'}. "
        "Properties: physicochemical, ADME heuristics, CYP inhibition, hERG risk, BBB penetration.\n"
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    doc.add_paragraph()
    fp = doc.add_paragraph(
        "DeepDock-AI  |  PyTorch + RDKit + AutoDock-Vina + GNINA  |  "
        f"Report generated {datetime.datetime.utcnow().strftime('%Y-%m-%d')}"
    )
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if fp.runs:
        fp.runs[0].font.size = Pt(8)
        fp.runs[0].font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _table_two_col(doc, rows: list, header_color: str = "0D47A1"):
    t = doc.add_table(rows=len(rows) + 1, cols=2)
    t.style = "Table Grid"
    for j, h in enumerate(("Metric", "Value")):
        c = t.rows[0].cells[j]
        c.text = h
        if c.paragraphs[0].runs:
            c.paragraphs[0].runs[0].bold = True
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _cell_bg(c, header_color)
    for i, (k, v) in enumerate(rows, start=1):
        t.rows[i].cells[0].text = str(k)
        t.rows[i].cells[1].text = str(v)
        if i % 2 == 0:
            for c in t.rows[i].cells:
                _cell_bg(c, "E3F2FD")


def _df_to_table(
    doc,
    df: pd.DataFrame,
    max_rows: int = 50,
    even_color: str = "F5F5F5",
    header_color: str = "1565C0",
    font_size: int = 8,
):
    """Convert DataFrame to DOCX table with proper NaN handling."""
    df = df.head(max_rows)
    if df.empty:
        doc.add_paragraph("No data.")
        return

    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    t.style = "Table Grid"

    for j, col in enumerate(df.columns):
        c = t.rows[0].cells[j]
        c.text = str(col)
        if c.paragraphs[0].runs:
            c.paragraphs[0].runs[0].bold = True
            c.paragraphs[0].runs[0].font.size = Pt(font_size)
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _cell_bg(c, header_color)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col in enumerate(df.columns):
            val = row[col]
            c = t.rows[i].cells[j]

            # ✅ FIX: Proper float/NaN handling
            if isinstance(val, float):
                if math.isfinite(val):
                    c.text = f"{val:.3f}"
                else:
                    c.text = "N/A"  # NaN, inf, -inf → "N/A"
            else:
                c.text = str(val) if val is not None else ""

            if c.paragraphs[0].runs:
                c.paragraphs[0].runs[0].font.size = Pt(font_size)
            if i % 2 == 0:
                _cell_bg(c, even_color)


def _bold_run(paragraph, text: str):
    run = paragraph.add_run(text)
    run.bold = True
