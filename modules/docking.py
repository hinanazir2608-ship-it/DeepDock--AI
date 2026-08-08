import os
import sys
import gc
import logging
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

import os
import subprocess
import urllib.request
import streamlit as st


@st.cache_resource
def get_vina_binary():
    """Download and set up standalone Vina binary if not present."""
    vina_path = "./vina"
    if not os.path.exists(vina_path):
        url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
        urllib.request.urlretrieve(url, vina_path)
        os.chmod(vina_path, 0o755)  # Make executable
    return vina_path


def run_docking(receptor_path, ligand_path, config_path, output_path):
    vina_bin = get_vina_binary()

    # Direct Linux CLI command call (Bypasses Python Memory Leakage)
    cmd = f"{vina_bin} --receptor {receptor_path} --ligand {ligand_path} --config {config_path} --out {output_path}"

    process = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if process.returncode == 0:
        return True, process.stdout
    else:
        return False, process.stderr
def load_gnina_model(model_path: Optional[str] = None) -> Any:
    if not TORCH_AVAILABLE:
        return None
    try:
        if model_path and os.path.exists(model_path):
            # Load model on CPU strictly
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
    exhaustiveness: int = 4,  # Memory safe limit
    n_poses: int = 5,        # Reduced pose memory load
    num_cpus: int = 1        # Restrict to 1 core to prevent Streamlit OOM crash
) -> Dict[str, Any]:
    
    if not VINA_AVAILABLE:
        raise RuntimeError("Vina module is not loaded properly.")

    results = {}

    for ligand_file in ligand_paths:
        try:
            # Pass cpu parameter to keep CPU usage low
            v = Vina(sf_name='vina', cpu=num_cpus)
            v.set_receptor(receptor_path)
            v.set_ligand_from_file(ligand_file)
            
            v.compute_vina_maps(center=center, box_size=box_size)
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            
            energies = v.energies(n_poses=n_poses)
            results[ligand_file] = {
                "status": "Success",
                "affinity": energies[0][0] if len(energies) > 0 else None,
                "all_poses_affinity": [e[0] for e in energies]
            }
        except Exception as e:
            results[ligand_file] = {"status": "Failed", "error": str(e)}
        finally:
            # Force Memory Cleanup immediately after each ligand
            gc.collect()

    return results
