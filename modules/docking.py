import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# --- Safe Imports with Graceful Fallbacks ---

# 1. Check AutoDock Vina Availability
try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError as e:
    VINA_AVAILABLE = False
    logger.warning(f"Vina module could not be imported: {e}")

# 2. Check OpenBabel / Meeko Availability
try:
    from openbabel import openbabel
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False

# 3. Check PyTorch / GNINA Model Availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --- Core Functions ---

def load_gnina_model(model_path: Optional[str] = None) -> Any:
    """
    Loads the GNINA scoring/docking model if PyTorch is installed.
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Cannot load GNINA model.")
        return None
    
    try:
        # Example model loading logic (adjust to your model architecture)
        if model_path and os.path.exists(model_path):
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()
            return model
        else:
            logger.warning("GNINA model path not found or not provided.")
            return None
    except Exception as e:
        logger.error(f"Error loading GNINA model: {e}")
        return None


def dock_ligands(
    receptor_path: str,
    ligand_paths: List[str],
    center: List[float],
    box_size: List[float],
    exhaustiveness: int = 8,
    n_poses: int = 9
) -> Dict[str, Any]:
    """
    Executes AutoDock Vina screening over a list of ligand PDBQT files.
    """
    if not VINA_AVAILABLE:
        raise RuntimeError(
            "AutoDock Vina is not installed or failed to load. "
            "Please check system Boost libraries (`libboost-all-dev`) in packages.txt."
        )

    results = {}

    for ligand_file in ligand_paths:
        try:
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_path)
            v.set_ligand_from_file(ligand_file)
            
            # Set grid box definition
            v.compute_vina_maps(center=center, box_size=box_size)
            
            # Run docking execution
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            
            # Retrieve energy scores
            energies = v.energies(n_poses=n_poses)
            results[ligand_file] = {
                "status": "Success",
                "affinity": energies[0][0] if len(energies) > 0 else None,
                "all_poses_affinity": [e[0] for e in energies]
            }
        except Exception as e:
            logger.error(f"Failed docking for {ligand_file}: {e}")
            results[ligand_file] = {
                "status": "Failed",
                "error": str(e)
            }

    return results
