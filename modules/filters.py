"""
Module 1: Parallel 2D ADMET Pre-Filtering
Lipinski RO5 + Veber + PAINS via RDKit multiprocessing
"""

import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Optional
import traceback

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams


# ── PAINS catalog (singleton per process) ────────────────────────────────────
def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


# ── Per-molecule filter worker (runs in subprocess) ───────────────────────────
def _filter_molecule(mol_smiles_name: Tuple[str, str]) -> Optional[dict]:
    """
    Returns a result dict if the molecule passes all filters, else None.
    Receives (SMILES, name) to avoid pickling RDKit Mol across processes.
    """
    smiles, name = mol_smiles_name
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mw    = Descriptors.ExactMolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        hbd   = rdMolDescriptors.CalcNumHBD(mol)
        hba   = rdMolDescriptors.CalcNumHBA(mol)
        rotb  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa  = rdMolDescriptors.CalcTPSA(mol)

        # ── Lipinski Rule of 5 ───────────────────────────────────────────────
        lipinski_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

        # ── Veber rules ──────────────────────────────────────────────────────
        veber_pass = (rotb <= 10 and tpsa <= 140)

        # ── PAINS ────────────────────────────────────────────────────────────
        catalog = _get_pains_catalog()
        pains_pass = not catalog.HasMatch(mol)

        violations = sum([
            mw > 500, logp > 5, hbd > 5, hba > 10, rotb > 10, tpsa > 140,
        ])

        return {
            "name":           name,
            "smiles":         smiles,
            "MW":             round(mw, 2),
            "LogP":           round(logp, 2),
            "HBD":            hbd,
            "HBA":            hba,
            "RotBonds":       rotb,
            "TPSA":           round(tpsa, 2),
            "LipinskiPass":   lipinski_pass,
            "VeberPass":      veber_pass,
            "PAINSPass":      pains_pass,
            "Violations":     violations,
            "AllPass":        lipinski_pass and veber_pass and pains_pass,
        }
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────
def run_parallel_admet_filter(
    mols: List[Chem.Mol],
    n_workers: Optional[int] = None,
) -> Tuple[List[Chem.Mol], List[dict]]:
    """
    Filter a list of RDKit Mol objects in parallel.

    Returns
    -------
    passed_mols : list[Mol]   — molecules passing all ADMET filters
    records     : list[dict]  — per-molecule stats (for every input mol)
    """
    if not mols:
        return [], []

    n_workers = n_workers or max(1, mp.cpu_count() - 1)

    # Convert mols → (SMILES, name) for pickle-safe IPC
    inputs: List[Tuple[str, str]] = []
    for i, mol in enumerate(mols):
        try:
            smi = Chem.MolToSmiles(mol)
            name = mol.GetPropsAsDict().get("_Name", f"mol_{i}")
        except Exception:
            smi, name = "", f"mol_{i}"
        inputs.append((smi, name))

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_filter_molecule, inputs)

    # Map results back to original mol objects
    passed_mols: List[Chem.Mol] = []
    records: List[dict] = []

    for mol, res in zip(mols, results):
        if res is None:
            continue
        records.append(res)
        if res["AllPass"]:
            passed_mols.append(mol)

    return passed_mols, records


def compute_single_admet(mol: Chem.Mol, name: str = "") -> dict:
    """Compute ADMET descriptors for a single molecule (no subprocess)."""
    smiles = Chem.MolToSmiles(mol)
    result = _filter_molecule((smiles, name or smiles[:20]))
    return result or {}
