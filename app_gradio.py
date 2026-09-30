"""
DeepDock-AI — app_gradio.py

GNINA rigid-receptor docking with a MANUAL active-site box.

Design notes:
  * No whole-protein centroid fallback. The user must supply the box, and the
    box is sanity-checked against the receptor (atoms inside the box).
  * GNINA CNN Score = pose confidence. For each ligand, the pose with the
    highest CNN Score is used for the reported scores and the complex PDB.
  * NO automatic ranking. Results are listed in input order; ranking is left
    to the user.
  * ADMET: ADMETlab 3.0 (via admetlab.py) when available, otherwise a built-in
    RDKit rule-based descriptor table.
  * Fixed GNINA / RDKit seeds. FORCE_CPU=True is the safest for repeatability.
    Bit-for-bit reproducibility still depends on identical GNINA/RDKit versions.
"""

import os
import re
import shutil
import zipfile
import tempfile
import subprocess
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from rdkit import Chem
from rdkit import __version__ as RDKIT_VERSION
from rdkit.Chem import Descriptors, Lipinski, AllChem, Crippen, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# ADMETlab 3.0 module (optional: the app still works with the RDKit fallback)
try:
    from admetlab import run_admet_analysis
    ADMETLAB_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001
    run_admet_analysis = None
    ADMETLAB_IMPORT_ERROR = str(_exc)


# ============================================================
# CONFIGURATION
# ============================================================

GNINA_SEED = 42
RDKIT_EMBED_SEED = 42

GNINA_NUM_MODES = 9
GNINA_EXHAUSTIVENESS = 8
GNINA_CPU = 4
GNINA_TIMEOUT = 1800

# True  -> always CPU (most repeatable)
# False -> GPU if an NVIDIA GPU is detected
FORCE_CPU = True

# Optional explicit GNINA path, e.g. r"C:\DeepDock-AI\gnina.exe"
GNINA_PATH = None

# Box fields start empty on purpose: the user must enter them.
DEFAULT_CENTER = None
DEFAULT_SIZE = 20.0

# Box must contain at least this many receptor atoms, otherwise it is
# almost certainly in empty space (e.g. left at 0,0,0).
MIN_ATOMS_IN_BOX = 50

# Modified residues stored as HETATM that should stay part of the protein.
MODIFIED_RESIDUES = {
    "MSE", "CSO", "SEP", "TPO", "PTR", "HYP", "KCX",
    "CME", "OCS", "CSD", "CAS", "MLY", "SMC",
}
WATER_NAMES = {"HOH", "WAT", "DOD"}


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_file_path(f):
    """Works with old Gradio file objects and new string paths."""
    if f is None:
        return None
    return getattr(f, "name", f)


def safe_filename(name):
    name = str(name).strip() or "ligand"
    return re.sub(r"[^\w\-.]", "_", name)[:80]


def is_valid_number(value):
    """Accept a legitimate 0.0 but reject None / NaN / text."""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


# ============================================================
# RECEPTOR PREPARATION
# ============================================================

def clean_receptor_pdb(pdb_file_path, output_path):
    """
    Protein-only receptor for rigid GNINA docking.

    Removes: waters, other HETATM (ligands, ions), altLoc B/C/..., extra models.
    Keeps:   ATOM records (altLoc blank/A, with the altLoc column blanked),
             common modified residues (re-labelled as ATOM), full PDB lines.
    Adds:    one TER between chains.

    Returns a dict with the residues that were converted / dropped so the
    user can verify nothing important near the active site was lost.
    """
    kept = []
    dropped_hetero = set()
    converted_modified = set()

    with open(pdb_file_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("ENDMDL"):
                break  # first model only

            is_atom = line.startswith("ATOM")
            is_het = line.startswith("HETATM")

            if not (is_atom or is_het) or len(line) < 54:
                continue

            resname = line[17:20].strip().upper()

            if is_het:
                if resname in MODIFIED_RESIDUES:
                    line = "ATOM  " + line[6:]
                    converted_modified.add(resname)
                else:
                    if resname not in WATER_NAMES:
                        dropped_hetero.add(resname)
                    continue

            # altLoc (column 17): keep blank / A only, then blank the column
            if line[16] not in (" ", "A"):
                continue
            line = line[:16] + " " + line[17:]

            kept.append(line)

    if not kept:
        raise ValueError("No protein ATOM records were found in the target PDB.")

    out = []
    previous_chain = None
    for line in kept:
        chain = line[21]
        if previous_chain is not None and chain != previous_chain:
            out.append("TER")
        out.append(line)
        previous_chain = chain
    out.append("TER")
    out.append("END")

    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")

    return {
        "atoms": len(kept),
        "converted_modified": sorted(converted_modified),
        "dropped_hetero": sorted(dropped_hetero),
    }


def load_protein_coords(pdb_path):
    coords = []
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") and len(line) >= 54:
                try:
                    coords.append([
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ])
                except ValueError:
                    continue
    return np.asarray(coords, dtype=float)


def count_atoms_in_box(coords, center, size):
    if coords.size == 0:
        return 0
    c = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    return int(np.all(np.abs(coords - c) <= half, axis=1).sum())


def get_native_ligand_center(pdb_path, resname):
    """Geometric centre of a native/co-crystal ligand (by 3-letter residue name)."""
    resname = (resname or "").strip().upper()
    coords = []

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM") or len(line) < 54:
                continue
            if line[16] not in (" ", "A"):
                continue
            r = line[17:20].strip().upper()
            if r in WATER_NAMES or r != resname:
                continue
            try:
                coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue

    if not coords:
        return None
    return tuple(np.mean(np.asarray(coords), axis=0))


def fill_center_from_native(target_file, resname):
    """UI helper: fill Center X/Y/Z from a native ligand in the uploaded PDB."""
    path = get_file_path(target_file)
    if not path:
        raise gr.Error("Upload the target PDB first.")
    resname = (resname or "").strip()
    if not resname:
        raise gr.Error("Enter the 3-letter residue name of the native ligand (e.g. from the HETATM lines).")

    center = get_native_ligand_center(path, resname)
    if center is None:
        raise gr.Error(f"No HETATM records named '{resname.upper()}' were found in the PDB.")

    return round(center[0], 3), round(center[1], 3), round(center[2], 3)


# ============================================================
# LIGAND 3D PREPARATION
# ============================================================

def prepare_ligand_3d(mol):
    """
    Returns (prepared_mol, ok, message).

    - Adds explicit H (with coordinates when a conformer exists).
    - Keeps an existing 3D conformer; otherwise deterministic ETKDGv3 embedding
      (random-coordinate retry if that fails).
    - MMFF (2000 its) -> UFF fallback; reports non-convergence honestly.
    """
    if mol is None:
        return None, False, "RDKit returned None."

    try:
        mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception as exc:
        return None, False, f"AddHs failed: {exc}"

    has_3d = False
    try:
        if mol.GetNumConformers() > 0:
            has_3d = mol.GetConformer().Is3D()
    except Exception:
        has_3d = False

    origin = "existing 3D conformer" if has_3d else "3D generated (ETKDGv3)"

    if not has_3d:
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = RDKIT_EMBED_SEED
            params.useRandomCoords = False
            conf_id = AllChem.EmbedMolecule(mol, params)

            if conf_id < 0:
                params = AllChem.ETKDGv3()
                params.randomSeed = RDKIT_EMBED_SEED
                params.useRandomCoords = True
                conf_id = AllChem.EmbedMolecule(mol, params)

            if conf_id < 0:
                return None, False, "3D embedding failed (deterministic and random-coordinate attempts)."
        except Exception as exc:
            return None, False, f"3D embedding failed: {exc}"

    # Force-field optimisation. MMFF returns 0 = converged, 1 = not converged,
    # -1 = could not set up (it does not necessarily raise).
    try:
        status = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        if status == 0:
            return mol, True, f"{origin}; MMFF converged."
        if status == 1:
            return mol, True, f"{origin}; MMFF did not fully converge."
    except Exception:
        pass

    try:
        status = AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        if status == 0:
            return mol, True, f"{origin}; UFF fallback converged."
        if status == 1:
            return mol, True, f"{origin}; UFF fallback did not fully converge."
    except Exception:
        pass

    return mol, True, f"{origin}; geometry optimisation unavailable."


# ============================================================
# LIGAND INPUT PROCESSING
# ============================================================

def find_smiles_column(df):
    for col in df.columns:
        if "smiles" in str(col).lower():
            return col
    return None


def find_name_column(df):
    lower_map = {str(c).lower(): c for c in df.columns}
    for key in ("name", "compound", "compound_name", "cid", "id"):
        if key in lower_map:
            return lower_map[key]
    return None


def _evaluate_ligand(mol, name, filter_type):
    """
    Prepare 3D and apply the selected filter.
    Returns (record, prepared_mol, rejection) - exactly one of
    (record & prepared_mol) or rejection is not None.
    """
    prepared, ok, prep_msg = prepare_ligand_3d(mol)
    if not ok:
        return None, None, {"Name": name, "Reason": prep_msg}

    try:
        mw = Descriptors.MolWt(prepared)
        logp = Descriptors.MolLogP(prepared)
        hbd = Lipinski.NumHDonors(prepared)
        hba = Lipinski.NumHAcceptors(prepared)
    except Exception as exc:
        return None, None, {"Name": name, "Reason": f"Descriptor calculation failed: {exc}"}

    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)

    ft = str(filter_type).lower()
    if ft == "lipinski" and violations > 0:
        return None, None, {"Name": name, "Reason": f"Failed Lipinski filter ({violations} violation(s))"}
    if ft.startswith("lipinski (1") and violations > 1:
        return None, None, {"Name": name, "Reason": f"Failed Lipinski filter ({violations} violations)"}

    prepared.SetProp("_Name", name)

    record = {
        "Name": name,
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": int(hbd),
        "HBA": int(hba),
        "Lipinski Violations": violations,
        "Preparation": prep_msg,
    }
    return record, prepared, None


def process_ligands(ligand_file_path, filter_type="Lipinski"):
    """Read CSV or SDF; every ligand gets 3D preparation."""
    ext = Path(ligand_file_path).suffix.lower()

    records, prepared_mols, rejected = [], [], []

    def handle(mol, name):
        rec, prepared, rej = _evaluate_ligand(mol, name, filter_type)
        if rej is not None:
            rejected.append(rej)
        else:
            records.append(rec)
            prepared_mols.append(prepared)

    if ext in (".sdf", ".sd"):
        supplier = Chem.SDMolSupplier(ligand_file_path, removeHs=False, sanitize=True)
        for i, mol in enumerate(supplier, start=1):
            if mol is None:
                rejected.append({"Name": f"SDF_{i}", "Reason": "Invalid SDF molecule"})
                continue
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            handle(mol, name or f"SDF_{i}")

    elif ext in (".csv", ".txt"):
        df = pd.read_csv(ligand_file_path)
        smiles_col = find_smiles_column(df)
        if smiles_col is None:
            raise ValueError("CSV must contain a column whose name includes 'SMILES'.")
        name_col = find_name_column(df)

        for i, row in df.iterrows():
            name = (
                str(row[name_col]).strip()
                if name_col is not None and pd.notna(row[name_col])
                else f"Mol_{i + 1}"
            )
            smiles = str(row[smiles_col]).strip()

            if not smiles or smiles.lower() == "nan":
                rejected.append({"Name": name, "Reason": "Missing SMILES"})
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rejected.append({"Name": name, "Reason": "Invalid SMILES"})
                continue

            handle(mol, name)
    else:
        raise ValueError("Supported ligand formats are CSV and SDF.")

    rejected_df = (
        pd.DataFrame(rejected) if rejected else pd.DataFrame(columns=["Name", "Reason"])
    )

    if not prepared_mols:
        raise ValueError(
            "No ligands remained after input validation/filtering "
            f"({len(rejected_df)} rejected). Check the rejection reasons or use filter 'None'."
        )

    return pd.DataFrame(records), prepared_mols, rejected_df


# ============================================================
# GNINA
# ============================================================

def get_gnina_path():
    candidates = []
    if GNINA_PATH:
        candidates.append(GNINA_PATH)

    for exe in ("gnina", "gnina.exe", "gninabase", "gninabase.exe"):
        found = shutil.which(exe)
        if found:
            candidates.append(found)

    app_dir = Path(__file__).resolve().parent
    for exe in ("gnina.exe", "gnina", "gninabase.exe", "gninabase"):
        candidates.append(str(app_dir / exe))

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())

    raise FileNotFoundError(
        "GNINA executable was not found. Add it to PATH or set GNINA_PATH."
    )


def get_gnina_version(gnina_exe):
    try:
        r = subprocess.run(
            [gnina_exe, "--version"], capture_output=True, text=True,
            timeout=20, errors="replace",
        )
        out = (r.stdout or r.stderr).strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


def detect_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True,
            timeout=10, errors="replace",
        )
        return r.returncode == 0
    except Exception:
        return False


def run_gnina_docking(
    gnina_exe, receptor_path, ligand_path, output_path,
    cx, cy, cz, sx, sy, sz, use_gpu,
):
    """Rigid-receptor GNINA docking in a manually specified box."""
    cmd = [
        gnina_exe,
        "-r", receptor_path,
        "-l", ligand_path,
        "-o", output_path,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--num_modes", str(GNINA_NUM_MODES),
        "--exhaustiveness", str(GNINA_EXHAUSTIVENESS),
        "--cnn_scoring", "rescore",
        "--seed", str(GNINA_SEED),
        "--cpu", str(GNINA_CPU),
    ]
    if not use_gpu:
        cmd.append("--no_gpu")

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=GNINA_TIMEOUT, errors="replace",
        )
        return _to_text(r.stdout), _to_text(r.stderr), r.returncode

    except subprocess.TimeoutExpired as exc:
        return (
            _to_text(exc.stdout),
            _to_text(exc.stderr) + f"\nGNINA timed out after {GNINA_TIMEOUT} s.",
            124,
        )
    except Exception as exc:  # noqa: BLE001
        return "", str(exc), 1


# ============================================================
# GNINA SCORE PARSING
# ============================================================

def _read_scores(mol):
    """Return (affinity, cnn_score, cnn_affinity) from one pose's SDF props."""
    normalized = {}
    for key in mol.GetPropNames():
        normalized[re.sub(r"[^a-z0-9]", "", key.lower())] = mol.GetProp(key)

    def get(*keys):
        for k in keys:
            if k in normalized:
                try:
                    return float(normalized[k])
                except (TypeError, ValueError):
                    return None
        return None

    return (
        get("minimizedaffinity", "affinity"),
        get("cnnscore"),
        get("cnnaffinity"),
    )


def parse_gnina_scores(gnina_output_path):
    """
    Read every pose and choose the one with the HIGHEST CNN Score
    (higher = better pose). Falls back to the first pose if none has a score.

    Returns (best_mol, affinity, cnn_score, cnn_affinity, n_poses)
    """
    supplier = Chem.SDMolSupplier(gnina_output_path, removeHs=False, sanitize=False)
    mols = [m for m in supplier if m is not None]

    if not mols:
        return None, None, None, None, 0

    best = None
    for m in mols:
        aff, cs, ca = _read_scores(m)
        key = cs if cs is not None else -np.inf
        if best is None or key > best[0]:
            best = (key, m, aff, cs, ca)

    _, mol, aff, cs, ca = best
    return mol, aff, cs, ca, len(mols)


# ============================================================
# POSE / COMPLEX
# ============================================================

def write_top_pose_sdf(mol, output_path):
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()
    return output_path


def ligand_mol_to_pdb(mol):
    try:
        return Chem.MolToPDBBlock(mol)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert docked ligand to PDB: {exc}")


def renumber_ligand_pdb(ligand_pdb_block, starting_serial):
    """
    Ligand ATOM/HETATM -> HETATM, residue LIG, chain Z, resid 1, with atom
    serials continuing after the receptor. Coordinate columns are preserved
    exactly (line[26:] starts at the iCode column).
    """
    out = []
    serial = int(starting_serial)

    for raw in ligand_pdb_block.splitlines():
        if not raw.startswith(("ATOM", "HETATM")) or len(raw) < 54:
            continue
        serial += 1
        out.append(f"HETATM{serial:5d}{raw[11:17]}LIG Z{1:4d}{raw[26:]}")

    return out, serial


def create_complex_pdb(receptor_pdb_path, ligand_pdb_block, output_path):
    """Protein + ligand with non-conflicting atom serials."""
    with open(receptor_pdb_path, "r", errors="ignore") as f:
        rec_lines = [
            ln.rstrip("\r\n") for ln in f if ln.startswith(("ATOM", "TER"))
        ]

    serials = []
    for ln in rec_lines:
        if ln.startswith("ATOM"):
            try:
                serials.append(int(ln[6:11]))
            except ValueError:
                pass
    max_serial = max(serials) if serials else 0

    ligand_lines, _ = renumber_ligand_pdb(ligand_pdb_block, starting_serial=max_serial)
    if not ligand_lines:
        raise RuntimeError("Docked ligand PDB block contained no atoms.")

    while rec_lines and rec_lines[-1] == "TER":
        rec_lines.pop()

    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(rec_lines) + "\n")
        f.write("TER\n")
        f.write("\n".join(ligand_lines) + "\n")
        f.write("TER\nEND\n")

    return output_path


# ============================================================
# ADMET: ADMETlab 3.0 -> RDKit fallback
# ============================================================

def _alert_catalogs():
    catalogs = {}
    try:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalogs["PAINS Alerts"] = FilterCatalog(p)
    except Exception:
        pass
    try:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalogs["Brenk Alerts"] = FilterCatalog(p)
    except Exception:
        pass
    return catalogs


def rdkit_admet_table(mols, names, affinities):
    """
    Built-in fallback. Rule-based physicochemical/drug-likeness descriptors
    computed with RDKit. NOT machine-learning ADMET predictions
    (no BBB / HIA / toxicity models).
    """
    catalogs = _alert_catalogs()
    rows = []

    for mol, name, aff in zip(mols, names, affinities):
        row = {"Name": name, "Docking Affinity (kcal/mol)": aff}
        try:
            m = Chem.RemoveHs(mol)
        except Exception:
            m = mol

        try:
            mw = Descriptors.MolWt(m)
            logp = Crippen.MolLogP(m)
            tpsa = rdMolDescriptors.CalcTPSA(m)
            hbd = rdMolDescriptors.CalcNumHBD(m)
            hba = rdMolDescriptors.CalcNumHBA(m)
            rotb = rdMolDescriptors.CalcNumRotatableBonds(m)

            row.update({
                "MW": round(mw, 2),
                "LogP (Crippen)": round(logp, 2),
                "TPSA": round(tpsa, 2),
                "HBD": int(hbd),
                "HBA": int(hba),
                "Rotatable Bonds": int(rotb),
                "Heavy Atoms": int(m.GetNumHeavyAtoms()),
                "Aromatic Rings": int(rdMolDescriptors.CalcNumAromaticRings(m)),
                "Fsp3": round(rdMolDescriptors.CalcFractionCSP3(m), 3),
                "Molar Refractivity": round(Crippen.MolMR(m), 2),
                "Lipinski Violations": int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10),
                "Veber Pass": "Yes" if (rotb <= 10 and tpsa <= 140) else "No",
                "Egan Pass": "Yes" if (logp <= 5.88 and tpsa <= 131.6) else "No",
            })

            try:
                row["QED"] = round(QED.qed(m), 3)
            except Exception:
                row["QED"] = np.nan

            for label, cat in catalogs.items():
                try:
                    row[label] = len(cat.GetMatches(m))
                except Exception:
                    row[label] = np.nan

            row["ADMET Source"] = "RDKit (rule-based descriptors)"

        except Exception as exc:  # noqa: BLE001
            row["ADMET Source"] = f"RDKit failed: {exc}"

        rows.append(row)

    return pd.DataFrame(rows)


def run_admet(mols, names, affinities):
    """
    Try ADMETlab 3.0 first; use RDKit whenever it is unavailable or fails.
    Returns (dataframe, source_label, log_message).
    """
    if not mols:
        return pd.DataFrame(), "none", "No successfully docked ligands for ADMET."

    reason = None

    if run_admet_analysis is None:
        reason = f"admetlab module not importable ({ADMETLAB_IMPORT_ERROR})"
    else:
        try:
            out = run_admet_analysis(
                mols=mols,
                names=names,
                scores=affinities,
                cids=names,
                use_api=True,
                status_text=None,
            )

            if isinstance(out, tuple):
                df = out[0]
                source = str(out[1]) if len(out) > 1 else "ADMETlab 3.0"
            else:
                df, source = out, "ADMETlab 3.0"

            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, source, f"ADMET source: {source}"

            reason = "ADMETlab returned an empty table"

        except Exception as exc:  # noqa: BLE001
            reason = f"ADMETlab call failed: {exc}"

    df = rdkit_admet_table(mols, names, affinities)
    return df, "RDKit", f"ADMETlab 3.0 unavailable ({reason}); RDKit fallback used."


# ============================================================
# MAIN PIPELINE (generator -> live progress)
# ============================================================

def docking_pipeline(
    ligand_file, filter_type, target_file,
    custom_cx, custom_cy, custom_cz,
    size_x, size_y, size_z,
):
    empty = pd.DataFrame()
    log = "🚀 Starting DeepDock-AI...\n"

    def out(lig=empty, dock=empty, rej=empty, admet=empty, csv=None, zp=None):
        return log, lig, dock, rej, admet, csv, zp

    # ---- validation
    ligand_path = get_file_path(ligand_file)
    target_path = get_file_path(target_file)

    if not ligand_path:
        raise gr.Error("Please upload a ligand CSV or SDF file.")
    if not target_path:
        raise gr.Error("Please upload a target protein PDB file.")

    centers = [custom_cx, custom_cy, custom_cz]
    sizes = [size_x, size_y, size_z]

    if not all(is_valid_number(v) for v in centers):
        raise gr.Error("Center X, Y and Z must all be filled in with numbers.")
    if not all(is_valid_number(v) and float(v) > 0 for v in sizes):
        raise gr.Error("Grid Size X/Y/Z must be positive numbers.")

    cx, cy, cz = (float(v) for v in centers)
    sx, sy, sz = (float(v) for v in sizes)

    log += (
        "\n🎯 MANUAL ACTIVE-SITE BOX"
        f"\n   Center: ({cx:.3f}, {cy:.3f}, {cz:.3f}) Å"
        f"\n   Size:   ({sx:.1f}, {sy:.1f}, {sz:.1f}) Å"
        "\n   Receptor: rigid"
    )
    yield out()

    run_dir = tempfile.mkdtemp(prefix="deepdock_ai_")       # working files (deleted)
    deliver_dir = tempfile.mkdtemp(prefix="deepdock_out_")  # final CSV/ZIP (kept for Gradio)

    try:
        receptor_clean = os.path.join(run_dir, "receptor_clean.pdb")
        docking_dir = os.path.join(run_dir, "docking")
        complexes_dir = os.path.join(run_dir, "complexes")
        poses_dir = os.path.join(run_dir, "top_poses")
        for d in (docking_dir, complexes_dir, poses_dir):
            os.makedirs(d, exist_ok=True)

        # ---- GNINA
        try:
            gnina_exe = get_gnina_path()
        except FileNotFoundError as exc:
            raise gr.Error(str(exc))

        gnina_version = get_gnina_version(gnina_exe)
        use_gpu = (not FORCE_CPU) and detect_gpu()

        log += (
            f"\n\n🔬 GNINA: {gnina_exe}"
            f"\n   Version: {gnina_version}"
            f"\n   Seed: {GNINA_SEED} | CPU threads: {GNINA_CPU} | "
            f"Exhaustiveness: {GNINA_EXHAUSTIVENESS} | Modes: {GNINA_NUM_MODES}"
            f"\n   Hardware: {'GPU' if use_gpu else 'CPU'}"
            "\n   CNN scoring: rescore"
            "\n   Pose used per ligand: highest CNN Score"
        )
        yield out()

        # ---- receptor
        log += "\n\n🧬 Preparing receptor..."
        try:
            info = clean_receptor_pdb(target_path, receptor_clean)
        except Exception as exc:
            raise gr.Error(f"Receptor preparation failed: {exc}")

        log += (
            f"\n   ✓ {info['atoms']} protein atoms kept (first model, altLoc blank/A)"
            "\n   ✓ Waters removed"
        )
        if info["dropped_hetero"]:
            log += f"\n   ℹ HETATM groups removed: {', '.join(info['dropped_hetero'])}"
        if info["converted_modified"]:
            log += f"\n   ℹ Modified residues kept as protein: {', '.join(info['converted_modified'])}"

        # ---- box sanity check
        coords = load_protein_coords(receptor_clean)
        n_in_box = count_atoms_in_box(coords, (cx, cy, cz), (sx, sy, sz))
        log += f"\n   ✓ Receptor atoms inside the box: {n_in_box}"

        if n_in_box < MIN_ATOMS_IN_BOX:
            raise gr.Error(
                f"The grid box contains only {n_in_box} receptor atoms "
                f"(minimum {MIN_ATOMS_IN_BOX}). The centre is probably outside the protein "
                "or left at 0,0,0. Check the coordinates."
            )
        yield out()

        # ---- ligands
        log += f"\n\n🧪 Preparing ligands (filter: {filter_type})..."
        try:
            ligand_df, prepared_mols, rejected_df = process_ligands(ligand_path, filter_type)
        except Exception as exc:
            raise gr.Error(f"Ligand preparation failed: {exc}")

        log += (
            f"\n   ✓ Accepted ligands: {len(prepared_mols)}"
            f"\n   ✓ Rejected before docking: {len(rejected_df)}"
        )
        yield out(lig=ligand_df, rej=rejected_df)

        # ---- docking loop
        docking_rows = []
        ok_mols, ok_names, ok_affinities = [], [], []
        total = len(prepared_mols)

        for idx, (mol, rec) in enumerate(zip(prepared_mols, ligand_df.to_dict("records")), start=1):
            name = str(rec["Name"])
            safe = safe_filename(name)

            log += f"\n\n[{idx}/{total}] 🔄 Docking: {name}"
            yield out(lig=ligand_df, dock=pd.DataFrame(docking_rows), rej=rejected_df)

            ligand_sdf = os.path.join(run_dir, f"{idx:03d}_{safe}.sdf")
            output_sdf = os.path.join(docking_dir, f"{idx:03d}_{safe}_docked.sdf")
            complex_pdb = os.path.join(complexes_dir, f"{idx:03d}_{safe}_complex.pdb")
            pose_sdf = os.path.join(poses_dir, f"{idx:03d}_{safe}_top_cnn_pose.sdf")

            try:
                writer = Chem.SDWriter(ligand_sdf)
                writer.write(mol)
                writer.close()

                _, stderr, rc = run_gnina_docking(
                    gnina_exe, receptor_clean, ligand_sdf, output_sdf,
                    cx, cy, cz, sx, sy, sz, use_gpu,
                )
                if rc != 0:
                    raise RuntimeError(f"GNINA exit code {rc}. {stderr[-800:]}")
                if not os.path.exists(output_sdf):
                    raise RuntimeError("GNINA produced no output SDF.")

                top, aff, cnn_score, cnn_aff, n_poses = parse_gnina_scores(output_sdf)
                if top is None:
                    raise RuntimeError("No valid docked pose found.")
                if cnn_score is None:
                    raise RuntimeError("CNN Score not found in GNINA output.")

                write_top_pose_sdf(top, pose_sdf)
                create_complex_pdb(receptor_clean, ligand_mol_to_pdb(top), complex_pdb)

                docking_rows.append({
                    "S.No": idx,
                    "Name": name,
                    "Affinity (kcal/mol)": round(aff, 3) if aff is not None else np.nan,
                    "CNN Score": round(cnn_score, 4),
                    "CNN Affinity": round(cnn_aff, 4) if cnn_aff is not None else np.nan,
                    "Poses Returned": n_poses,
                    "Docking Status": "Success",
                })

                ok_mols.append(mol)
                ok_names.append(name)
                ok_affinities.append(aff if aff is not None else np.nan)

                log += (
                    "\n   ✓ Success"
                    f"\n   Affinity: {aff if aff is not None else 'N/A'} kcal/mol"
                    f" | CNN Score: {cnn_score:.4f}"
                    f" | CNN Affinity: {cnn_aff if cnn_aff is not None else 'N/A'}"
                )

            except Exception as exc:  # noqa: BLE001
                docking_rows.append({
                    "S.No": idx,
                    "Name": name,
                    "Affinity (kcal/mol)": np.nan,
                    "CNN Score": np.nan,
                    "CNN Affinity": np.nan,
                    "Poses Returned": 0,
                    "Docking Status": f"Failed: {exc}",
                })
                log += f"\n   ✗ Failed: {exc}"

            yield out(lig=ligand_df, dock=pd.DataFrame(docking_rows), rej=rejected_df)

        results_df = pd.DataFrame(docking_rows)

        # ---- ADMET (successful dockings only)
        log += f"\n\n🧬 ADMET for {len(ok_mols)} successfully docked ligand(s)..."
        yield out(lig=ligand_df, dock=results_df, rej=rejected_df)

        admet_df, admet_source, admet_msg = run_admet(ok_mols, ok_names, ok_affinities)
        log += f"\n   {'✓' if admet_source != 'none' else 'ℹ'} {admet_msg}"
        if admet_source == "RDKit":
            log += "\n   ℹ RDKit table = rule-based descriptors (no ML BBB/HIA/toxicity models)."

        # ---- files
        ligands_csv = os.path.join(deliver_dir, "DeepDockAI_Prepared_Ligands.csv")
        docking_csv = os.path.join(deliver_dir, "DeepDockAI_Docking_Results.csv")
        rejected_csv = os.path.join(deliver_dir, "DeepDockAI_Rejected_Ligands.csv")
        admet_csv = os.path.join(deliver_dir, "DeepDockAI_ADMET_Results.csv")

        ligand_df.to_csv(ligands_csv, index=False)
        results_df.to_csv(docking_csv, index=False)
        rejected_df.to_csv(rejected_csv, index=False)
        admet_df.to_csv(admet_csv, index=False)

        params_txt = os.path.join(run_dir, "run_parameters.txt")
        with open(params_txt, "w", encoding="utf-8") as f:
            f.write(
                "DeepDock-AI run parameters\n"
                f"GNINA: {gnina_exe}\nGNINA version: {gnina_version}\n"
                f"GNINA seed: {GNINA_SEED}\nGNINA CPU threads: {GNINA_CPU}\n"
                f"GNINA exhaustiveness: {GNINA_EXHAUSTIVENESS}\nGNINA num_modes: {GNINA_NUM_MODES}\n"
                f"Hardware: {'GPU' if use_gpu else 'CPU'}\nCNN scoring: rescore\n"
                "Pose used: highest CNN Score\n"
                f"RDKit version: {RDKIT_VERSION}\nRDKit embed seed: {RDKIT_EMBED_SEED}\n"
                f"Ligand filter: {filter_type}\n"
                f"Grid center: {cx}, {cy}, {cz}\nGrid size: {sx}, {sy}, {sz}\n"
                f"Receptor atoms in box: {n_in_box}\n"
                f"ADMET source: {admet_source}\n"
                f"Python: {platform.python_version()} on {platform.system()}\n"
            )

        zip_path = os.path.join(deliver_dir, "DeepDockAI_Results.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in (ligands_csv, docking_csv, rejected_csv, admet_csv):
                zf.write(p, arcname=os.path.basename(p))
            zf.write(params_txt, arcname="run_parameters.txt")
            zf.write(receptor_clean, arcname="receptor_clean.pdb")
            for folder, arc in ((complexes_dir, "complexes"), (poses_dir, "top_poses")):
                for root, _, files in os.walk(folder):
                    for fn in files:
                        zf.write(os.path.join(root, fn), arcname=os.path.join(arc, fn))

        log += (
            "\n\n✅ DeepDock-AI completed."
            "\n   Results are listed in input order (no automatic ranking)."
            "\n   Failed dockings were not sent to ADMET."
        )
        yield out(ligand_df, results_df, rejected_df, admet_df, docking_csv, zip_path)

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="DeepDock-AI — GNINA Rigid Docking") as demo:

    gr.Markdown(
        """
# 🧬 DeepDock-AI

### GNINA rigid-receptor docking with manual active-site coordinates

- Manual grid centre and size (checked against the receptor before docking)
- Rigid receptor, GNINA docking, CNN rescoring
- Pose reported per ligand = highest **CNN Score** (pose confidence)
- Results are **not ranked automatically**
- Fixed seeds; CPU mode is the default for repeatability
- ADMET: ADMETlab 3.0 when available, otherwise an RDKit fallback
"""
    )

    with gr.Row():
        ligand_file = gr.File(label="Ligands (.csv / .sdf)", file_types=[".csv", ".sdf", ".sd"])
        target_file = gr.File(label="Target Protein (.pdb)", file_types=[".pdb"])

    filter_type = gr.Dropdown(
        choices=["Lipinski", "Lipinski (1 violation allowed)", "None"],
        value="Lipinski",
        label="Ligand Filtering",
    )

    gr.Markdown("## 🎯 Manual Active-Site / Grid Definition")

    with gr.Row():
        custom_cx = gr.Number(label="Center X (Å)", value=DEFAULT_CENTER)
        custom_cy = gr.Number(label="Center Y (Å)", value=DEFAULT_CENTER)
        custom_cz = gr.Number(label="Center Z (Å)", value=DEFAULT_CENTER)

    with gr.Row():
        size_x = gr.Number(label="Grid Size X (Å)", value=DEFAULT_SIZE, minimum=1)
        size_y = gr.Number(label="Grid Size Y (Å)", value=DEFAULT_SIZE, minimum=1)
        size_z = gr.Number(label="Grid Size Z (Å)", value=DEFAULT_SIZE, minimum=1)

    with gr.Row():
        native_resname = gr.Textbox(
            label="Native ligand residue name (optional helper)",
            placeholder="3-letter code from the HETATM lines of the PDB",
        )
        fill_btn = gr.Button("📍 Fill centre from native ligand")

    gr.Markdown(
        "Use the same box for every compound in a comparative run. "
        "The pipeline stops if the box contains almost no receptor atoms."
    )

    submit_btn = gr.Button("🚀 Run GNINA Docking", variant="primary")

    status_box = gr.Textbox(label="Live Progress", lines=18, interactive=False)

    ligand_table = gr.Dataframe(label="Prepared Ligands", interactive=False)
    docking_table = gr.Dataframe(label="GNINA Docking Results", interactive=False)
    rejected_table = gr.Dataframe(label="Rejected / Skipped Ligands", interactive=False)
    admet_table = gr.Dataframe(label="ADMET Results", interactive=False)

    docking_csv_file = gr.File(label="Download Docking CSV")
    results_zip_file = gr.File(label="Download Complete Results ZIP")

    fill_btn.click(
        fn=fill_center_from_native,
        inputs=[target_file, native_resname],
        outputs=[custom_cx, custom_cy, custom_cz],
    )

    submit_btn.click(
        fn=docking_pipeline,
        inputs=[
            ligand_file, filter_type, target_file,
            custom_cx, custom_cy, custom_cz,
            size_x, size_y, size_z,
        ],
        outputs=[
            status_box,
            ligand_table, docking_table, rejected_table, admet_table,
            docking_csv_file, results_zip_file,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
