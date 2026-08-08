"""
Stage 1 — Parallel ADMET Pre-Filter
Lipinski RO5 + Veber oral bioavailability + PAINS A/B/C, run via multiprocessing.Pool.

Public interface (matches app.py imports):
    run_parallel_admet_filter(mols) -> (passed_mols, records)
    compute_single_admet(mol)       -> record dict for one molecule
"""

from __future__ import annotations
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

# Build the PAINS catalog once per process (expensive to construct).
_PAINS_CATALOG: Optional[FilterCatalog] = None


def _get_pains_catalog() -> FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


def compute_single_admet(mol: Chem.Mol) -> Dict:
    """
    Runs Lipinski + Veber + PAINS on one molecule.
    Returns a flat dict — safe to pickle across multiprocessing workers.
    Mirrors the naming used in apply_rdkit_filters() from the Gradio version.
    """
    name = "Unnamed"
    if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        name = mol.GetProp("_Name").strip()

    if mol is None:
        return {
            "Name": name, "LipinskiPass": False, "VeberPass": False,
            "PAINSPass": False, "AllPass": False, "Error": "Invalid molecule",
        }

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Lipinski Rule of 5 — allow ≤1 violation, same tolerance as the
        # original apply_rdkit_filters() Lipinski branch.
        lipinski_viols = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lipinski_pass = lipinski_viols <= 1

        # Veber oral bioavailability rule
        veber_pass = rotb <= 10 and tpsa <= 140

        # PAINS A/B/C
        pains_hit = _get_pains_catalog().HasMatch(mol)
        pains_pass = not pains_hit

        return {
            "Name": name,
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "HBD": hbd,
            "HBA": hba,
            "TPSA": round(tpsa, 2),
            "RotBonds": rotb,
            "LipinskiViolations": lipinski_viols,
            "LipinskiPass": lipinski_pass,
            "VeberPass": veber_pass,
            "PAINSPass": pains_pass,
            "AllPass": lipinski_pass and veber_pass and pains_pass,
        }
    except Exception as e:
        return {
            "Name": name, "LipinskiPass": False, "VeberPass": False,
            "PAINSPass": False, "AllPass": False, "Error": str(e),
        }


def _worker(mol_bytes_and_props: Tuple[bytes, dict]) -> Dict:
    """Pool worker — reconstructs the Mol from a pickled binary + re-attaches props."""
    mol_bytes, props = mol_bytes_and_props
    try:
        mol = Chem.Mol(mol_bytes)
        for k, v in props.items():
            mol.SetProp(k, str(v))
        return compute_single_admet(mol)
    except Exception as e:
        return {"Name": props.get("_Name", "Unknown"), "LipinskiPass": False,
                "VeberPass": False, "PAINSPass": False, "AllPass": False, "Error": str(e)}


def run_parallel_admet_filter(
    mols: List[Chem.Mol],
    n_workers: Optional[int] = None,
) -> Tuple[List[Chem.Mol], List[Dict]]:
    """
    Filters `mols` in parallel using multiprocessing.Pool.

    Returns
    -------
    passed : list of RDKit Mol objects that passed all three checks
    records : list of per-molecule filter-result dicts (for the Stage 1 report table)
    """
    if not mols:
        return [], []

    n_workers = n_workers or max(1, min(mp.cpu_count() - 1, 8))

    # Molecules aren't directly picklable across process boundaries in all
    # RDKit builds without care, so we serialise to binary + carry the name
    # separately. This also matches how the SDF loader already stamps _Name.
    payload = []
    for mol in mols:
        try:
            props = {"_Name": mol.GetProp("_Name")} if mol.HasProp("_Name") else {}
            payload.append((mol.ToBinary(), props))
        except Exception:
            payload.append((b"", {"_Name": "Unnamed"}))

    if len(mols) < 20 or n_workers <= 1:
        # Small sets: skip Pool overhead, run in-process.
        records = [compute_single_admet(m) for m in mols]
    else:
        with mp.Pool(processes=n_workers) as pool:
            records = pool.map(_worker, payload)

    passed = [mol for mol, rec in zip(mols, records) if rec.get("AllPass")]
    return passed, records
