"""
DeepDock-AI — docking.py

GNINA rigid-receptor docking module designed to match app_gradio.py.

Key behavior:
  - Uses GNINA, not AutoDock Vina.
  - Manual active-site/grid coordinates are required.
  - Uses the same grid for every ligand in a comparative run.
  - GNINA generates 9 poses per ligand and uses CNN Score to select the
    reported pose, matching app_gradio.py.
  - Reports GNINA affinity, CNN Score and CNN Affinity.
  - Automatically uses an NVIDIA GPU when detected; otherwise CPU.
  - Fixed GNINA/RDKit seeds for reproducibility.
  - Deterministic RDKit ligand preparation when a 3D conformer is absent.
  - No whole-protein centroid/origin fallback.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import __version__ as RDKIT_VERSION
from rdkit.Chem import AllChem


# ============================================================
# SETTINGS — kept aligned with app_gradio.py
# ============================================================

GNINA_SEED = 42
RDKIT_EMBED_SEED = 42
GNINA_NUM_MODES = 9
GNINA_EXHAUSTIVENESS = 8
GNINA_CPU = 4
GNINA_TIMEOUT = 1800

# Automatically use GPU when an NVIDIA GPU is available.
# If False, --no_gpu is passed to GNINA and CPU is used.
FORCE_CPU = False

# Optional explicit executable path.
GNINA_PATH = None

DEFAULT_GRID_SIZE = (20.0, 20.0, 20.0)
MIN_ATOMS_IN_BOX = 50

MODIFIED_RESIDUES = {
    "MSE", "CSO", "SEP", "TPO", "PTR", "HYP", "KCX",
    "CME", "OCS", "CSD", "CAS", "MLY", "SMC",
}
WATER_NAMES = {"HOH", "WAT", "DOD"}


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_file_path(f):
    """Accept a Gradio file object or a normal filesystem path."""
    if f is None:
        return None
    return getattr(f, "name", f)


def safe_filename(name):
    name = str(name).strip() or "ligand"
    return re.sub(r"[^\w\-.]", "_", name)[:80]


def is_valid_number(value):
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# ============================================================
# GNINA DETECTION / VERSION / GPU
# ============================================================

def get_gnina_path():
    """Locate GNINA from explicit path, PATH, or beside the application."""
    candidates = []

    if GNINA_PATH:
        candidates.append(GNINA_PATH)

    for exe in ("gnina", "gnina.exe", "gninabase", "gninabase.exe"):
        found = shutil.which(exe)
        if found:
            candidates.append(found)

    try:
        app_dir = Path(__file__).resolve().parent
        for exe in ("gnina.exe", "gnina", "gninabase.exe", "gninabase"):
            candidates.append(str(app_dir / exe))
    except Exception:
        pass

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())

    raise FileNotFoundError(
        "GNINA executable was not found. Add GNINA to PATH or set GNINA_PATH."
    )


def get_gnina_version(gnina_exe=None):
    if gnina_exe is None:
        try:
            gnina_exe = get_gnina_path()
        except Exception:
            return "unknown"

    try:
        result = subprocess.run(
            [gnina_exe, "--version"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        out = (result.stdout or result.stderr or "").strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


def detect_gpu():
    """Return True only when nvidia-smi confirms an NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def select_hardware(force_cpu=None):
    """Return (use_gpu, label). GPU is selected automatically unless forced off."""
    if force_cpu is None:
        force_cpu = FORCE_CPU

    use_gpu = (not bool(force_cpu)) and detect_gpu()
    return use_gpu, ("GPU" if use_gpu else "CPU")


# ============================================================
# RECEPTOR PREPARATION / GRID VALIDATION
# ============================================================

def clean_receptor_pdb(pdb_file_path, output_path):
    """
    Prepare a rigid protein-only receptor for GNINA.

    Keeps ATOM records and common modified residues represented as HETATM.
    Removes waters, ordinary HETATM groups and alternate conformers other
    than blank/A. Only the first model is retained.
    """
    kept = []
    dropped_hetero = set()
    converted_modified = set()

    with open(pdb_file_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("ENDMDL"):
                break

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

            if line[16] not in (" ", "A"):
                continue

            # Blank alternate-location column after selecting A/blank.
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

    out.extend(["TER", "END"])

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


def validate_grid(grid_center, grid_size=DEFAULT_GRID_SIZE, receptor_pdb=None):
    """Validate manual box and optionally require enough receptor atoms inside."""
    if grid_center is None or len(grid_center) != 3:
        return False, "grid_center must contain exactly 3 coordinates.", 0

    if grid_size is None or len(grid_size) != 3:
        return False, "grid_size must contain exactly 3 dimensions.", 0

    try:
        center = tuple(float(x) for x in grid_center)
        size = tuple(float(x) for x in grid_size)
    except (TypeError, ValueError):
        return False, "Grid coordinates and dimensions must be numeric.", 0

    if not all(np.isfinite(x) for x in center + size):
        return False, "Grid coordinates and dimensions must be finite.", 0

    if any(x <= 0 for x in size):
        return False, "Grid dimensions must be greater than zero.", 0

    if receptor_pdb is not None:
        coords = load_protein_coords(receptor_pdb)
        n_in_box = count_atoms_in_box(coords, center, size)
        if n_in_box < MIN_ATOMS_IN_BOX:
            return False, (
                f"The grid box contains only {n_in_box} receptor atoms "
                f"(minimum {MIN_ATOMS_IN_BOX}). Check the active-site coordinates."
            ), n_in_box
        return True, None, n_in_box

    return True, None, 0


def get_native_ligand_center(pdb_path, resname):
    """Return the geometric center of a native/co-crystal HETATM ligand."""
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


# ============================================================
# LIGAND 3D PREPARATION
# ============================================================

def prepare_ligand_3d(mol):
    """Prepare ligand with existing 3D coordinates or deterministic ETKDG."""
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
                return None, False, "3D embedding failed."
        except Exception as exc:
            return None, False, f"3D embedding failed: {exc}"

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


def write_ligand_sdf(mol, output_path):
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()
    return output_path


# ============================================================
# GNINA DOCKING
# ============================================================

def run_gnina_docking(
    gnina_exe,
    receptor_path,
    ligand_path,
    output_path,
    cx, cy, cz,
    sx, sy, sz,
    use_gpu,
):
    """Run rigid-receptor GNINA docking with CNN rescoring."""
    cmd = [
        gnina_exe,
        "-r", receptor_path,
        "-l", ligand_path,
        "-o", output_path,
        "--center_x", str(cx),
        "--center_y", str(cy),
        "--center_z", str(cz),
        "--size_x", str(sx),
        "--size_y", str(sy),
        "--size_z", str(sz),
        "--num_modes", str(GNINA_NUM_MODES),
        "--exhaustiveness", str(GNINA_EXHAUSTIVENESS),
        "--cnn_scoring", "rescore",
        "--seed", str(GNINA_SEED),
        "--cpu", str(GNINA_CPU),
    ]

    if not use_gpu:
        cmd.append("--no_gpu")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=GNINA_TIMEOUT,
            errors="replace",
        )
        return _to_text(result.stdout), _to_text(result.stderr), result.returncode
    except subprocess.TimeoutExpired as exc:
        return (
            _to_text(exc.stdout),
            _to_text(exc.stderr) + f"\nGNINA timed out after {GNINA_TIMEOUT} s.",
            124,
        )
    except Exception as exc:
        return "", str(exc), 1


# ============================================================
# GNINA SCORE PARSING
# ============================================================

def _read_scores(mol):
    """Read affinity, CNN Score and CNN Affinity from one GNINA SDF pose."""
    normalized = {}
    for key in mol.GetPropNames():
        normalized[re.sub(r"[^a-z0-9]", "", key.lower())] = mol.GetProp(key)

    def get(*keys):
        for key in keys:
            if key in normalized:
                try:
                    return float(normalized[key])
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
    Parse all returned GNINA poses and select the pose with the HIGHEST
    CNN Score, matching app_gradio.py.

    Returns:
        best_mol, affinity, cnn_score, cnn_affinity, n_poses
    """
    supplier = Chem.SDMolSupplier(
        gnina_output_path,
        removeHs=False,
        sanitize=False,
    )
    mols = [m for m in supplier if m is not None]

    if not mols:
        return None, None, None, None, 0

    best = None
    for mol in mols:
        affinity, cnn_score, cnn_affinity = _read_scores(mol)
        key = cnn_score if cnn_score is not None else -np.inf
        if best is None or key > best[0]:
            best = (key, mol, affinity, cnn_score, cnn_affinity)

    _, mol, affinity, cnn_score, cnn_affinity = best
    return mol, affinity, cnn_score, cnn_affinity, len(mols)


def write_top_pose_sdf(mol, output_path):
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()
    return output_path


# ============================================================
# COMPLEX CREATION
# ============================================================

def ligand_mol_to_pdb(mol):
    try:
        return Chem.MolToPDBBlock(mol)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert docked ligand to PDB: {exc}")


def renumber_ligand_pdb(ligand_pdb_block, starting_serial):
    """Convert ligand atoms to HETATM and assign residue LIG on chain Z."""
    out = []
    serial = int(starting_serial)

    for raw in ligand_pdb_block.splitlines():
        if not raw.startswith(("ATOM", "HETATM")) or len(raw) < 54:
            continue
        serial += 1
        out.append(
            f"HETATM{serial:5d}{raw[11:17]}LIG Z{1:4d}{raw[26:]}"
        )

    return out, serial


def create_complex_pdb(receptor_pdb_path, ligand_pdb_block, output_path):
    """Write receptor + docked ligand with non-conflicting atom serials."""
    with open(receptor_pdb_path, "r", errors="ignore") as f:
        rec_lines = [
            ln.rstrip("\r\n")
            for ln in f
            if ln.startswith(("ATOM", "TER"))
        ]

    serials = []
    for line in rec_lines:
        if line.startswith("ATOM"):
            try:
                serials.append(int(line[6:11]))
            except ValueError:
                pass

    max_serial = max(serials) if serials else 0
    ligand_lines, _ = renumber_ligand_pdb(
        ligand_pdb_block,
        starting_serial=max_serial,
    )

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
# SINGLE-LIGAND API
# ============================================================

def dock_single_ligand(
    mol,
    idx,
    receptor_pdb,
    grid_center,
    grid_size=DEFAULT_GRID_SIZE,
    output_dir=None,
    gnina_path=None,
    use_gpu=None,
):
    """
    Dock one prepared RDKit ligand.

    Returns a dict compatible with the GNINA result table used by app_gradio.py.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="deepdock_gnina_")
    os.makedirs(output_dir, exist_ok=True)

    if gnina_path is None:
        gnina_path = get_gnina_path()

    if use_gpu is None:
        use_gpu, _ = select_hardware()

    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol_{idx + 1}"
    safe = safe_filename(name)

    ligand_sdf = os.path.join(output_dir, f"{idx + 1:03d}_{safe}.sdf")
    output_sdf = os.path.join(output_dir, f"{idx + 1:03d}_{safe}_docked.sdf")
    pose_sdf = os.path.join(output_dir, f"{idx + 1:03d}_{safe}_top_cnn_pose.sdf")
    complex_pdb = os.path.join(output_dir, f"{idx + 1:03d}_{safe}_complex.pdb")

    result = {
        "S.No": idx + 1,
        "Name": name,
        "Affinity (kcal/mol)": np.nan,
        "CNN Score": np.nan,
        "CNN Affinity": np.nan,
        "Poses Returned": 0,
        "Docking Status": "",
        "Docked SDF": None,
        "Top Pose SDF": None,
        "Complex PDB": None,
    }

    try:
        write_ligand_sdf(mol, ligand_sdf)

        stdout, stderr, rc = run_gnina_docking(
            gnina_path,
            receptor_pdb,
            ligand_sdf,
            output_sdf,
            grid_center[0], grid_center[1], grid_center[2],
            grid_size[0], grid_size[1], grid_size[2],
            use_gpu,
        )

        if rc != 0:
            raise RuntimeError(f"GNINA exit code {rc}. {stderr[-800:]}")
        if not os.path.exists(output_sdf):
            raise RuntimeError("GNINA produced no output SDF.")

        top, affinity, cnn_score, cnn_affinity, n_poses = parse_gnina_scores(
            output_sdf
        )

        if top is None:
            raise RuntimeError("No valid docked pose found.")
        if cnn_score is None:
            raise RuntimeError("CNN Score not found in GNINA output.")

        write_top_pose_sdf(top, pose_sdf)
        create_complex_pdb(
            receptor_pdb,
            ligand_mol_to_pdb(top),
            complex_pdb,
        )

        result.update({
            "Affinity (kcal/mol)": round(affinity, 3) if affinity is not None else np.nan,
            "CNN Score": round(cnn_score, 4),
            "CNN Affinity": round(cnn_affinity, 4) if cnn_affinity is not None else np.nan,
            "Poses Returned": n_poses,
            "Docking Status": "Success",
            "Docked SDF": output_sdf,
            "Top Pose SDF": pose_sdf,
            "Complex PDB": complex_pdb,
        })
        return result

    except Exception as exc:
        result["Docking Status"] = f"Failed: {exc}"
        return result


# ============================================================
# BATCH API
# ============================================================

def run_batched_docking(
    filtered_mols,
    target_pdb_file,
    grid_center,
    grid_size=DEFAULT_GRID_SIZE,
    output_dir=None,
    progress=None,
    force_cpu=None,
):
    """
    GNINA batch docking API.

    Parameters intentionally mirror the docking responsibility of app_gradio.py:
      filtered_mols : prepared RDKit molecules
      target_pdb_file : PDB path or Gradio file object
      grid_center : (x, y, z), manually supplied active-site center
      grid_size : (x, y, z), in Angstrom
      progress : optional Gradio progress callback
      force_cpu : override automatic GPU detection for this run

    Returns a list of dictionaries with Affinity, CNN Score and CNN Affinity.
    """
    total = len(filtered_mols)
    if total == 0:
        return []

    target_path = get_file_path(target_pdb_file)
    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError("Target protein PDB was not found.")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="deepdock_gnina_")
    os.makedirs(output_dir, exist_ok=True)

    if not grid_center or len(grid_center) != 3:
        raise ValueError("grid_center must contain exactly three coordinates.")
    if not grid_size or len(grid_size) != 3:
        raise ValueError("grid_size must contain exactly three dimensions.")

    cx = tuple(float(x) for x in grid_center)
    size = tuple(float(x) for x in grid_size)

    receptor_clean = os.path.join(output_dir, "receptor_clean.pdb")
    info = clean_receptor_pdb(target_path, receptor_clean)

    ok, error, n_in_box = validate_grid(
        cx,
        size,
        receptor_pdb=receptor_clean,
    )
    if not ok:
        raise ValueError(error)

    gnina_path = get_gnina_path()
    use_gpu, hardware = select_hardware(force_cpu)
    gnina_version = get_gnina_version(gnina_path)

    print(f"GNINA: {gnina_path}")
    print(f"GNINA version: {gnina_version}")
    print(f"Hardware: {hardware}")
    print(f"GNINA seed: {GNINA_SEED}")
    print(f"CPU threads: {GNINA_CPU}")
    print(f"Exhaustiveness: {GNINA_EXHAUSTIVENESS}")
    print(f"Num modes: {GNINA_NUM_MODES}")
    print(f"CNN scoring: rescore")
    print(f"Grid center: {cx}")
    print(f"Grid size: {size}")
    print(f"Receptor atoms in box: {n_in_box}")
    print(f"Receptor atoms kept: {info['atoms']}")

    results = []
    for idx, mol in enumerate(filtered_mols):
        if progress is not None:
            progress(
                (idx + 1) / total,
                desc=f"GNINA docking [{idx + 1}/{total}]",
            )

        result = dock_single_ligand(
            mol=mol,
            idx=idx,
            receptor_pdb=receptor_clean,
            grid_center=cx,
            grid_size=size,
            output_dir=output_dir,
            gnina_path=gnina_path,
            use_gpu=use_gpu,
        )
        results.append(result)

        if result["Docking Status"] == "Success":
            print(
                f"[{idx + 1}/{total}] {result['Name']} | "
                f"Affinity={result['Affinity (kcal/mol)']} | "
                f"CNN Score={result['CNN Score']} | "
                f"CNN Affinity={result['CNN Affinity']}"
            )
        else:
            print(f"[{idx + 1}/{total}] {result['Name']} | {result['Docking Status']}")

    return results


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

# app_gradio.py uses these names directly; keeping them here makes the module
# easy to import without changing the Gradio pipeline.
parse_scores = parse_gnina_scores
run_docking = run_gnina_docking


if __name__ == "__main__":
    print("DeepDock-AI GNINA docking module")
    try:
        path = get_gnina_path()
        use_gpu, hardware = select_hardware()
        print(f"GNINA: {path}")
        print(f"Version: {get_gnina_version(path)}")
        print(f"Hardware selected: {hardware}")
    except Exception as exc:
        print(f"GNINA detection failed: {exc}")
