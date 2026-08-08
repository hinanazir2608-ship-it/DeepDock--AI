import os
import sys
import gc
import re
import logging
import subprocess
import urllib.request
import tempfile
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Safe imports
try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_vina_binary() -> str:
    """Download and set up standalone Vina binary if Python bindings fail."""
    vina_path = os.path.join(tempfile.gettempdir(), "vina_standalone")
    if not os.path.exists(vina_path):
        try:
            url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
            urllib.request.urlretrieve(url, vina_path)
            os.chmod(vina_path, 0o755)
        except Exception as e:
            logger.error(f"Failed to download Vina binary: {e}")
    return vina_path


def sanitize_pdbqt_file(file_path: str) -> bool:
    """
    In-place sanitizer: Ensures no illegal tags (COMPND, HEADER, REMARK, etc.)
    reach Vina's C++ parser.
    """
    if not os.path.exists(file_path):
        return False

    ALLOWED_KEYWORDS = ("ATOM", "HETATM", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        sanitized_lines = [
            line for line in lines 
            if line.strip().startswith(ALLOWED_KEYWORDS)
        ]

        if not sanitized_lines:
            return False

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(sanitized_lines)
            if not sanitized_lines[-1].endswith("\n"):
                f.write("\n")

        return True
    except Exception as e:
        logger.error(f"Sanitization failed for {file_path}: {e}")
        return False


def load_gnina_model(model_path: Optional[str] = None) -> Any:
    if not TORCH_AVAILABLE:
        return None
    try:
        if model_path and os.path.exists(model_path):
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()
            return model
        return None
    except Exception as e:
        logger.error(f"GNINA model load failed: {e}")
        return None


def dock_ligands(
    receptor_path: str,
    ligand_paths: List[str],
    center: List[float],
    box_size: List[float],
    exhaustiveness: int = 4,
    n_poses: int = 5,
    num_cpus: int = 1
) -> Dict[str, Any]:
    
    # 1. Sanitize Receptor
    sanitize_pdbqt_file(receptor_path)
    
    results = {}

    for ligand_file in ligand_paths:
        # 2. Sanitize Ligand PDBQT before feeding to Vina
        if not sanitize_pdbqt_file(ligand_file):
            results[ligand_file] = {
                "status": "Failed",
                "error": "PDBQT parsing error: No valid ATOM/HETATM records found after sanitization."
            }
            continue

        docked_success = False

        # --- Method A: Python Vina Bindings ---
        if VINA_AVAILABLE:
            try:
                v = Vina(sf_name='vina', cpu=num_cpus)
                v.set_receptor(receptor_path)
                v.set_ligand_from_file(ligand_file)
                
                v.compute_vina_maps(center=center, box_size=box_size)
                v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
                
                energies = v.energies(n_poses=n_poses)
                results[ligand_file] = {
                    "status": "Success",
                    "affinity": float(energies[0][0]) if len(energies) > 0 else 0.0,
                    "all_poses_affinity": [float(e[0]) for e in energies]
                }
                docked_success = True
            except Exception as e:
                logger.warning(f"Python Vina binding failed for {ligand_file}: {e}. Retrying via CLI binary...")

        # --- Method B: Fallback Standalone Vina CLI ---
        if not docked_success:
            try:
                vina_bin = get_vina_binary()
                out_path = ligand_file.replace(".pdbqt", "_out.pdbqt")

                cmd = [
                    vina_bin,
                    "--receptor", receptor_path,
                    "--ligand", ligand_file,
                    "--center_x", str(center[0]),
                    "--center_y", str(center[1]),
                    "--center_z", str(center[2]),
                    "--size_x", str(box_size[0]),
                    "--size_y", str(box_size[1]),
                    "--size_z", str(box_size[2]),
                    "--exhaustiveness", str(exhaustiveness),
                    "--num_modes", str(n_poses),
                    "--cpu", str(num_cpus),
                    "--out", out_path
                ]

                process = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                if process.returncode == 0:
                    # Parse affinity score from output string
                    affinity = None
                    for line in process.stdout.splitlines():
                        if line.strip().startswith("1"):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    affinity = float(parts[1])
                                    break
                                except ValueError:
                                    pass

                    results[ligand_file] = {
                        "status": "Success",
                        "affinity": affinity if affinity is not None else 0.0,
                        "all_poses_affinity": [affinity] if affinity is not None else []
                    }
                else:
                    results[ligand_file] = {
                        "status": "Failed",
                        "error": process.stderr.strip() or "CLI Vina Execution Failed"
                    }

            except Exception as e:
                results[ligand_file] = {"status": "Failed", "error": str(e)}

        # Memory Cleanup
        gc.collect()

    return results
