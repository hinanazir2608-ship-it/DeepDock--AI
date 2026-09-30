"""
DeepDock-AI — docking.py

Reproducible AutoDock Vina docking module (Meeko-based ligand preparation).

Reproducibility measures:
  1. Fixed RDKit ETKDG seed for ligand 3D embedding.
  2. Fixed Vina random seed, CPU count and exhaustiveness.
  3. Ligand PDBQT is generated with Meeko (proper AutoDock atom types,
     Gasteiger charges, torsion tree) instead of hand-written PDBQT lines.
  4. Vina affinity is read from the output PDBQT REMARK VINA RESULT line.
  5. No fabricated/fallback docking scores.
  6. Vina executable is detected from VINA_PATH env var or PATH.
  7. Vina / Meeko / RDKit versions are logged before docking.
  8. Grid center and size are validated (no silent docking at the origin).
  9. Each ligand gets its own uniquely-named files (index prefix).
 10. Every Vina call has a timeout.

NOTE: Pin your versions (vina, meeko, rdkit) and report them in the Methods.
"""

import os
import re
import shutil
import subprocess
import tempfile

from rdkit import Chem
from rdkit import __version__ as RDKIT_VERSION
from rdkit.Chem import AllChem


# ============================================================
# REPRODUCIBILITY / RUN SETTINGS
# ============================================================

RDKIT_EMBED_SEED = 42
VINA_SEED = 42

# Keep identical on every machine being compared.
VINA_CPU = 4
VINA_EXHAUSTIVENESS = 8
VINA_NUM_MODES = 9

# Per-ligand timeout (seconds)
VINA_TIMEOUT = 1800

# Default grid dimensions (Angstrom)
DEFAULT_GRID_SIZE = (20.0, 20.0, 20.0)

# Optional: set env var VINA_PATH to a full path of the vina executable.


# ============================================================
# TOOL DETECTION / VERSIONS
# ============================================================

def get_vina_path():
    """Locate AutoDock Vina: VINA_PATH env var first, then PATH."""
    env_path = os.environ.get("VINA_PATH")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    for name in ("vina", "vina.exe"):
        found = shutil.which(name)
        if found:
            return os.path.abspath(found)

    return None


def get_vina_version(vina_path=None):
    """Return the Vina version string (or None)."""
    if vina_path is None:
        vina_path = get_vina_path()
    if vina_path is None:
        return None

    try:
        result = subprocess.run(
            [vina_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (result.stdout or result.stderr).strip()
        return version if version else None
    except Exception:
        return None


def get_meeko_version():
    """Return the installed Meeko version, or None if Meeko is missing."""
    try:
        import meeko
        return getattr(meeko, "__version__", "unknown")
    except Exception:
        return None


def get_obabel_path():
    """Locate Open Babel (used only as a fallback / for PDB export)."""
    for name in ("obabel", "obabel.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


# ============================================================
# NAME HELPERS
# ============================================================

def get_mol_name(mol, idx):
    """Return the molecule name, or a generic Ligand_N."""
    if mol is not None and mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name
    return f"Ligand_{idx + 1}"


def make_safe_name(name):
    """Filesystem-safe name (works on Windows/Linux)."""
    safe = re.sub(r"[^\w\-.]", "_", str(name))
    return safe[:80] if safe else "ligand"


# ============================================================
# LIGAND 3D PREPARATION
# ============================================================

def prepare_deterministic_3d_molecule(mol):
    """
    Create a deterministic 3D ligand conformer with explicit hydrogens.

    Returns:
        (prepared_mol, note, error_message)
        note is an optional non-fatal warning (e.g. undefined stereo).
    """
    notes = []

    try:
        if mol is None:
            return None, None, "Invalid RDKit molecule."

        # Work on a copy so the original is untouched.
        mol = Chem.Mol(mol)

        # Warn about undefined stereocenters (ETKDG will pick one arbitrarily).
        try:
            centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False
            )
            if any(label == "?" for _, label in centers):
                notes.append("undefined stereocenter(s): arbitrary isomer embedded")
        except Exception:
            pass

        mol = Chem.AddHs(mol)
        mol.RemoveAllConformers()

        # Attempt 1: deterministic ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = RDKIT_EMBED_SEED
        params.useRandomCoords = False
        result = AllChem.EmbedMolecule(mol, params)

        # Attempt 2: random coords (still seeded, so still deterministic)
        if result == -1:
            params = AllChem.ETKDGv3()
            params.randomSeed = RDKIT_EMBED_SEED
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)

        if result == -1:
            return None, None, "RDKit 3D embedding failed."

        # Force-field optimisation
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                status = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
                if status == 1:
                    notes.append("MMFF did not fully converge")
            else:
                status = AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
                notes.append("UFF used (no MMFF params)")
                if status == 1:
                    notes.append("UFF did not fully converge")
        except Exception:
            notes.append("force-field optimisation failed; raw embedded coords used")

        return mol, "; ".join(notes) if notes else None, None

    except Exception as e:
        return None, None, f"3D preparation error: {e}"


# ============================================================
# LIGAND -> PDBQT (Meeko)
# ============================================================

def _meeko_pdbqt_string(mol_3d):
    """Generate a PDBQT string with Meeko. Returns (string, error)."""
    try:
        from meeko import MoleculePreparation
    except Exception as e:
        return None, f"Meeko not installed ({e}). Install with: pip install meeko"

    try:
        preparator = MoleculePreparation()

        # Newer Meeko (>=0.5): prepare() returns a list of MoleculeSetup
        try:
            from meeko import PDBQTWriterLegacy

            setups = preparator.prepare(mol_3d)
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
            if not is_ok:
                return None, f"Meeko PDBQT writing failed: {err}"
            return pdbqt_string, None

        except ImportError:
            # Older Meeko API
            preparator.prepare(mol_3d)
            return preparator.write_pdbqt_string(), None

    except Exception as e:
        return None, f"Meeko error: {e}"


def _obabel_pdbqt(mol_3d, output_pdbqt_path):
    """Fallback: RDKit mol -> SDF -> obabel PDBQT. Returns (ok, error)."""
    obabel = get_obabel_path()
    if obabel is None:
        return False, "Open Babel not found for fallback conversion."

    tmp_sdf = output_pdbqt_path + ".tmp.sdf"
    try:
        writer = Chem.SDWriter(tmp_sdf)
        writer.write(mol_3d)
        writer.close()

        proc = subprocess.run(
            [obabel, tmp_sdf, "-O", output_pdbqt_path, "-xh"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not os.path.exists(output_pdbqt_path):
            return False, f"obabel failed: {proc.stderr.strip()[:300]}"
        return True, None
    except Exception as e:
        return False, f"obabel error: {e}"
    finally:
        if os.path.exists(tmp_sdf):
            try:
                os.remove(tmp_sdf)
            except OSError:
                pass


def mol_to_pdbqt(mol, output_pdbqt_path, allow_obabel_fallback=False):
    """
    Convert an RDKit molecule into a valid Vina PDBQT file.

    Returns:
        (success, note, error)
    """
    try:
        output_dir = os.path.dirname(output_pdbqt_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        prepared_mol, note, error = prepare_deterministic_3d_molecule(mol)
        if prepared_mol is None:
            return False, None, error

        pdbqt_string, meeko_err = _meeko_pdbqt_string(prepared_mol)

        if pdbqt_string:
            with open(output_pdbqt_path, "w", encoding="utf-8") as f:
                f.write(pdbqt_string)
            return True, note, None

        if allow_obabel_fallback:
            ok, ob_err = _obabel_pdbqt(prepared_mol, output_pdbqt_path)
            if ok:
                extra = "Open Babel PDBQT used (Meeko failed)"
                return True, f"{note}; {extra}" if note else extra, None
            return False, note, f"{meeko_err} | {ob_err}"

        return False, note, meeko_err

    except Exception as e:
        return False, None, f"Error generating PDBQT: {e}"


# ============================================================
# RECEPTOR PREPARATION (optional helpers)
# ============================================================

def clean_receptor_pdb(input_pdb, output_pdb):
    """
    Keep protein ATOM records only:
      - drops waters and HETATM
      - keeps only altLoc ' ' or 'A'
      - keeps full ATOM lines (element columns intact)
      - inserts TER between chains
    """
    cleaned = []
    current_chain = None

    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break  # first model only

            if not line.startswith("ATOM") or len(line) < 54:
                continue

            if line[16] not in (" ", "A"):
                continue

            chain = line[21]
            if current_chain is not None and chain != current_chain:
                cleaned.append("TER\n")
            current_chain = chain

            cleaned.append(line if line.endswith("\n") else line + "\n")

    if not cleaned:
        return False

    cleaned.append("TER\nEND\n")
    with open(output_pdb, "w", encoding="utf-8") as f:
        f.writelines(cleaned)
    return True


def prepare_receptor_pdbqt(input_pdb, output_pdbqt):
    """
    Convert a receptor PDB to PDBQT (adds hydrogens + charges).

    Tries Meeko's mk_prepare_receptor.py first (CLI flags differ between Meeko
    versions, so both styles are attempted), then Open Babel as a fallback.
    ALWAYS inspect the result once and check the catalytic residues
    (e.g. His41 / Cys145 for Mpro) look correct.

    Returns: (success, tool_used_or_error)
    """
    clean_pdb = os.path.splitext(output_pdbqt)[0] + "_clean.pdb"
    if not clean_receptor_pdb(input_pdb, clean_pdb):
        return False, "No protein ATOM records found after cleaning."

    base = os.path.splitext(output_pdbqt)[0]

    mk = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    if mk:
        attempts = [
            [mk, "--read_pdb", clean_pdb, "-o", base, "--write_pdbqt", output_pdbqt],
            [mk, "-i", clean_pdb, "-o", base, "-p"],
        ]
        for cmd in attempts:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if proc.returncode == 0 and os.path.exists(output_pdbqt):
                    return True, "Meeko mk_prepare_receptor"
                # some versions name the file <base>.pdbqt
                alt = base + ".pdbqt"
                if proc.returncode == 0 and os.path.exists(alt):
                    if alt != output_pdbqt:
                        shutil.move(alt, output_pdbqt)
                    return True, "Meeko mk_prepare_receptor"
            except Exception:
                continue

    obabel = get_obabel_path()
    if obabel:
        try:
            proc = subprocess.run(
                [obabel, clean_pdb, "-O", output_pdbqt, "-xr", "-h"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode == 0 and os.path.exists(output_pdbqt):
                return True, "Open Babel (fallback)"
        except Exception:
            pass

    return False, "No receptor preparation tool worked (Meeko / Open Babel)."


# ============================================================
# OUTPUT PARSING
# ============================================================

def parse_affinity_from_pdbqt(out_pdbqt_path):
    """Extract the top-pose Vina affinity from `REMARK VINA RESULT:`."""
    if not os.path.exists(out_pdbqt_path):
        return None

    try:
        with open(out_pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            return float(parts[3])
                        except ValueError:
                            return None
        return None
    except Exception:
        return None


def extract_top_pose_pdbqt(out_pdbqt_path, top_pdbqt_path):
    """Write only MODEL 1 of a Vina output file. Returns bool."""
    if not os.path.exists(out_pdbqt_path):
        return False

    lines = []
    in_model = False
    try:
        with open(out_pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MODEL"):
                    if in_model:
                        break
                    in_model = True
                    continue
                if line.startswith("ENDMDL"):
                    break
                lines.append(line)

        if not lines:
            return False

        with open(top_pdbqt_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True
    except Exception:
        return False


def convert_pdbqt_to_pdb(pdbqt_path, pdb_path):
    """
    Convert a docked pose PDBQT to PDB with Open Babel (optional).
    Returns bool. Bond orders are NOT preserved in PDBQT/PDB; use it for
    visualisation / complex building, not for chemistry.
    """
    obabel = get_obabel_path()
    if obabel is None or not os.path.exists(pdbqt_path):
        return False
    try:
        proc = subprocess.run(
            [obabel, pdbqt_path, "-O", pdb_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode == 0 and os.path.exists(pdb_path)
    except Exception:
        return False


# ============================================================
# GRID VALIDATION
# ============================================================

def validate_grid(grid_center, grid_size):
    """Validate docking grid parameters. Returns (ok, error)."""
    if grid_center is None or len(grid_center) != 3:
        return False, "grid_center must contain exactly 3 coordinates."

    if grid_size is None or len(grid_size) != 3:
        return False, "grid_size must contain exactly 3 dimensions."

    try:
        center = tuple(float(x) for x in grid_center)
        size = tuple(float(x) for x in grid_size)
    except (TypeError, ValueError):
        return False, "Grid coordinates and dimensions must be numeric."

    if any(x <= 0 for x in size):
        return False, "Grid dimensions must be greater than zero."

    if center == (0.0, 0.0, 0.0):
        return False, (
            "grid_center is (0,0,0). Provide the actual binding-site "
            "coordinates (e.g. centre of the native ligand or catalytic residues)."
        )

    return True, None


# ============================================================
# SINGLE LIGAND DOCKING
# ============================================================

def dock_single_ligand(
    mol,
    idx,
    target_pdbqt_file,
    grid_center,
    grid_size,
    output_dir,
    vina_path,
    allow_obabel_fallback=False,
):
    """
    Dock one ligand.

    Returns a dict:
        affinity, status, note, ligand_pdbqt, output_pdbqt, top_pose_pdbqt
    """
    mol_name = get_mol_name(mol, idx)
    safe_name = f"{idx + 1:04d}_{make_safe_name(mol_name)}"

    ligand_pdbqt = os.path.join(output_dir, f"{safe_name}.pdbqt")
    output_pdbqt = os.path.join(output_dir, f"{safe_name}_out.pdbqt")
    top_pose_pdbqt = os.path.join(output_dir, f"{safe_name}_top1.pdbqt")

    result = {
        "affinity": None,
        "status": "",
        "note": None,
        "ligand_pdbqt": ligand_pdbqt,
        "output_pdbqt": output_pdbqt,
        "top_pose_pdbqt": None,
    }

    # Remove stale outputs so old results are never reused.
    for path in (ligand_pdbqt, output_pdbqt, top_pose_pdbqt):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # ---- Ligand preparation
    converted, note, error = mol_to_pdbqt(
        mol, ligand_pdbqt, allow_obabel_fallback=allow_obabel_fallback
    )
    result["note"] = note

    if not converted:
        result["status"] = f"Failed: {error}"
        return result

    # ---- Vina command
    vina_cmd = [
        vina_path,
        "--receptor", target_pdbqt_file,
        "--ligand", ligand_pdbqt,
        "--center_x", str(grid_center[0]),
        "--center_y", str(grid_center[1]),
        "--center_z", str(grid_center[2]),
        "--size_x", str(grid_size[0]),
        "--size_y", str(grid_size[1]),
        "--size_z", str(grid_size[2]),
        "--out", output_pdbqt,
        "--exhaustiveness", str(VINA_EXHAUSTIVENESS),
        "--num_modes", str(VINA_NUM_MODES),
        "--seed", str(VINA_SEED),
        "--cpu", str(VINA_CPU),
    ]

    # ---- Run Vina
    try:
        process = subprocess.run(
            vina_cmd,
            capture_output=True,
            text=True,
            timeout=VINA_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        result["status"] = f"Failed: Vina timed out after {VINA_TIMEOUT} s"
        return result
    except Exception as e:
        result["status"] = f"Failed: could not execute Vina: {e}"
        return result

    if process.returncode != 0:
        stderr = (process.stderr or process.stdout or "Unknown Vina error.").strip()
        result["status"] = f"Failed: Vina exited {process.returncode}: {stderr[:500]}"
        return result

    # ---- Affinity
    affinity = parse_affinity_from_pdbqt(output_pdbqt)

    if affinity is None:
        result["status"] = "Failed: Vina completed but no REMARK VINA RESULT was found."
        return result

    if extract_top_pose_pdbqt(output_pdbqt, top_pose_pdbqt):
        result["top_pose_pdbqt"] = top_pose_pdbqt

    result["affinity"] = affinity
    result["status"] = "Docked Successfully"
    return result


# ============================================================
# BATCH DOCKING
# ============================================================

def _fail_all(filtered_mols, message):
    """Build a failed-result row for every ligand."""
    rows = []
    for idx, mol in enumerate(filtered_mols):
        rows.append({
            "S.No": idx + 1,
            "Molecule Name": get_mol_name(mol, idx),
            "Affinity (kcal/mol)": None,
            "Status": message,
            "Note": None,
            "Output PDBQT": None,
            "Top Pose PDBQT": None,
        })
    return rows


def run_batched_docking(
    filtered_mols,
    target_pdbqt_file,
    grid_center=None,
    grid_size=DEFAULT_GRID_SIZE,
    output_dir=None,
    progress=None,
    allow_obabel_fallback=False,
):
    """
    Execute reproducible AutoDock Vina docking.

    Args:
        filtered_mols: list of RDKit molecules
        target_pdbqt_file: receptor .pdbqt (path or file object with .name)
        grid_center: (x, y, z) of the binding site (required, must not be 0,0,0)
        grid_size: (x, y, z) in Angstrom
        output_dir: folder for outputs (temp folder created if None)
        progress: optional Gradio progress callback
        allow_obabel_fallback: use Open Babel PDBQT if Meeko fails
                               (Meeko is strongly preferred; the fallback
                               should be reported in Methods if it is ever used)

    Returns:
        list of dicts, one per ligand.
    """
    total = len(filtered_mols)
    if total == 0:
        return []

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="deepdock_vina_")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Locate Vina
    vina_path = get_vina_path()
    if vina_path is None:
        msg = "Failed: Vina binary not found (PATH or VINA_PATH)"
        print(f"❌ {msg}")
        return _fail_all(filtered_mols, msg)

    # ---- Meeko check
    meeko_version = get_meeko_version()
    if meeko_version is None and not (allow_obabel_fallback and get_obabel_path()):
        msg = "Failed: Meeko not installed (pip install meeko)"
        print(f"❌ {msg}")
        return _fail_all(filtered_mols, msg)

    vina_version = get_vina_version(vina_path)

    print(f"ℹ️ Vina executable: {vina_path}")
    print(f"ℹ️ Vina version: {vina_version if vina_version else 'Unknown'}")
    print(f"ℹ️ Meeko version: {meeko_version if meeko_version else 'not installed'}")
    print(f"ℹ️ RDKit version: {RDKIT_VERSION}")
    print(f"ℹ️ Vina seed: {VINA_SEED} | CPU: {VINA_CPU} | "
          f"exhaustiveness: {VINA_EXHAUSTIVENESS} | modes: {VINA_NUM_MODES}")
    print(f"ℹ️ RDKit embedding seed: {RDKIT_EMBED_SEED}")

    # ---- Validate receptor
    target_path = (
        target_pdbqt_file.name
        if hasattr(target_pdbqt_file, "name")
        else str(target_pdbqt_file)
    )

    if not os.path.exists(target_path):
        msg = f"Failed: receptor PDBQT not found at {target_path}"
        print(f"❌ {msg}")
        return _fail_all(filtered_mols, msg)

    if not target_path.lower().endswith(".pdbqt"):
        msg = ("Failed: receptor must be PDBQT "
               "(use prepare_receptor_pdbqt() to convert a PDB first)")
        print(f"❌ {msg}")
        return _fail_all(filtered_mols, msg)

    # ---- Validate grid
    grid_ok, grid_error = validate_grid(grid_center, grid_size)
    if not grid_ok:
        print(f"❌ Invalid docking grid: {grid_error}")
        return _fail_all(filtered_mols, f"Failed: {grid_error}")

    print(f"ℹ️ Grid center: {grid_center[0]}, {grid_center[1]}, {grid_center[2]}")
    print(f"ℹ️ Grid size:   {grid_size[0]}, {grid_size[1]}, {grid_size[2]}")

    # ---- Dock each ligand
    results = []

    for idx, mol in enumerate(filtered_mols):
        mol_name = get_mol_name(mol, idx)

        if progress is not None:
            progress(
                0.2 + 0.6 * (idx + 1) / total,
                desc=f"⚡ Docking [{idx + 1}/{total}]: {mol_name}",
            )

        print(f"\n🔬 Docking [{idx + 1}/{total}]: {mol_name}")

        res = dock_single_ligand(
            mol=mol,
            idx=idx,
            target_pdbqt_file=target_path,
            grid_center=grid_center,
            grid_size=grid_size,
            output_dir=output_dir,
            vina_path=vina_path,
            allow_obabel_fallback=allow_obabel_fallback,
        )

        results.append({
            "S.No": idx + 1,
            "Molecule Name": mol_name,
            "Affinity (kcal/mol)": res["affinity"],
            "Status": res["status"],
            "Note": res["note"],
            "Output PDBQT": res["output_pdbqt"],
            "Top Pose PDBQT": res["top_pose_pdbqt"],
        })

        if res["affinity"] is not None:
            extra = f"  [{res['note']}]" if res["note"] else ""
            print(f"   ✔ Affinity: {res['affinity']:.3f} kcal/mol{extra}")
        else:
            print(f"   ⚠️ {res['status']}")

    # ---- Summary
    successful = sum(1 for r in results if r["Affinity (kcal/mol)"] is not None)
    failed = total - successful

    print("\n======================================")
    print("DOCKING COMPLETE")
    print("======================================")
    print(f"Total ligands:  {total}")
    print(f"Successful:     {successful}")
    print(f"Failed:         {failed}")
    print(f"Vina version:   {vina_version}")
    print(f"Meeko version:  {meeko_version}")
    print(f"RDKit version:  {RDKIT_VERSION}")
    print(f"Vina seed:      {VINA_SEED}")
    print(f"Vina CPU:       {VINA_CPU}")
    print(f"Exhaustiveness: {VINA_EXHAUSTIVENESS}")
    print(f"Output dir:     {output_dir}")
    print("======================================")

    return results
