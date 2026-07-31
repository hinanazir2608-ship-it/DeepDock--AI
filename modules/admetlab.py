"""
ADMETlab 3.0 Integration
API client for https://admetlab3.scbdd.com + RDKit fallback
Reference: Gui C. et al. Nucleic Acids Research 2024 (ADMETlab 3.0)
"""

from __future__ import annotations
import time
import math
from typing import List, Dict, Optional

import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, AllChem


# ── ADMETlab 3.0 API ──────────────────────────────────────────────────────────
ADMETLAB_BASE  = "https://admetlab3.scbdd.com"
SUBMIT_ENDPOINT = f"{ADMETLAB_BASE}/api/predict"
RESULT_ENDPOINT = f"{ADMETLAB_BASE}/api/result"

# Properties to request from ADMETlab 3.0
ADMET_TASKS = [
    # Physicochemical
    "MW", "logP", "logD", "HBD", "HBA", "TPSA", "RotBonds",
    "nRings", "nArRings", "Fsp3",
    # ADME
    "Caco2",        "HIA", "Pgp_inh", "Pgp_sub",
    "BBB",          "CNS_MPO",
    "F_20", "F_30",
    "T_12",         "VD",
    "CL_hep",       "CL_micro",
    # Metabolism
    "CYP1A2_inh", "CYP1A2_sub",
    "CYP2C9_inh", "CYP2C9_sub",
    "CYP2C19_inh","CYP2C19_sub",
    "CYP2D6_inh", "CYP2D6_sub",
    "CYP3A4_inh", "CYP3A4_sub",
    # Toxicity
    "AMES",         "hERG",
    "H_HT",         "DILI",
    "Skin_Sen",     "Respiratory_Tox",
    "Carcinogens_Ames",
    # Drug-likeness
    "QED", "SA_Score", "NP_Score", "Lipinski", "Pfizer_Rule",
    "GSK_Rule", "PAINS",
]


def _submit_smiles(smiles_list: List[str], timeout: int = 30) -> Optional[str]:
    """Submit SMILES to ADMETlab 3.0; returns task_id or None."""
    payload = {
        "smiles": "\n".join(smiles_list),
        "models": ADMET_TASKS,
    }
    try:
        r = requests.post(SUBMIT_ENDPOINT, json=payload, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return data.get("task_id") or data.get("id")
    except Exception:
        pass
    return None


def _poll_results(task_id: str, max_wait: int = 120) -> Optional[List[Dict]]:
    """Poll ADMETlab 3.0 for results; returns list of property dicts."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{RESULT_ENDPOINT}/{task_id}", timeout=20)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "")
                if status == "done":
                    return data.get("results", [])
                elif status in ("error", "failed"):
                    return None
        except Exception:
            pass
        time.sleep(3)
    return None


def predict_via_admetlab3(
    smiles_list: List[str],
    names: List[str],
    status_text=None,
) -> Optional[pd.DataFrame]:
    """
    Submit SMILES to ADMETlab 3.0 and return results DataFrame.
    Returns None if the API is unreachable or times out.
    """
    if status_text:
        status_text.info("Submitting to ADMETlab 3.0 API…")

    task_id = _submit_smiles(smiles_list)
    if not task_id:
        return None

    if status_text:
        status_text.info(f"ADMETlab 3.0 task submitted (id={task_id[:8]}…). Waiting…")

    raw = _poll_results(task_id)
    if not raw:
        return None

    records = []
    for i, entry in enumerate(raw):
        row: Dict = {"Name": names[i] if i < len(names) else f"cpd_{i+1}",
                     "SMILES": smiles_list[i] if i < len(smiles_list) else ""}
        for prop, val in entry.items():
            if prop not in ("smiles", "id"):
                row[prop] = val
        records.append(row)

    return pd.DataFrame(records)


# ── RDKit ADMET Fallback ──────────────────────────────────────────────────────
def _esol_logs(mol: Chem.Mol) -> float:
    """Delaney ESOL model for aqueous solubility estimation (LogS)."""
    try:
        mw    = Descriptors.MolWt(mol)
        clogp = Descriptors.MolLogP(mol)
        rotb  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_ar  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        frac  = n_ar / max(mol.GetNumHeavyAtoms(), 1)
        return round(0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rotb - 0.74 * frac, 3)
    except Exception:
        return float("nan")


def _sa_score_rdkit(mol: Chem.Mol) -> float:
    """Synthetic Accessibility Score (Ertl & Schuffenhauer 2009) via RDKit."""
    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        return round(sascorer.calculateScore(mol), 3)
    except Exception:
        return float("nan")


def _bbb_heuristic(mol: Chem.Mol) -> str:
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return "Yes" if (mw < 450 and 0 < logp < 5 and hbd <= 3 and tpsa < 90) else "No"


def _herg_risk(mol: Chem.Mol) -> str:
    logp = Descriptors.MolLogP(mol)
    mw   = Descriptors.MolWt(mol)
    # Simple structural alert heuristic
    smarts_herg = Chem.MolFromSmarts("[#7;!$(N-C=O)]~[#6]~[#6]~[c]")
    has_alert = mol.HasSubstructMatch(smarts_herg) if smarts_herg else False
    if logp > 4.5 and has_alert:
        return "High Risk"
    elif logp > 3.5:
        return "Medium Risk"
    return "Low Risk"


def _drug_likeness_label(qed: float, viols: int) -> str:
    if qed >= 0.7 and viols == 0:  return "Excellent"
    if qed >= 0.5 and viols <= 1:  return "Good"
    if qed >= 0.3:                  return "Moderate"
    return "Poor"


def predict_rdkit_admet(
    mols:   List[Chem.Mol],
    names:  List[str],
    scores: List[float],
) -> pd.DataFrame:
    """
    Comprehensive ADMET profiling using RDKit descriptors as fallback.
    Covers physicochemical, ADME heuristics, and flag-based toxicity.
    """
    records = []
    for mol, name, score in zip(mols, names, scores):
        try:
            mw      = round(Descriptors.ExactMolWt(mol), 2)
            logp    = round(Descriptors.MolLogP(mol), 2)
            hbd     = rdMolDescriptors.CalcNumHBD(mol)
            hba     = rdMolDescriptors.CalcNumHBA(mol)
            tpsa    = round(rdMolDescriptors.CalcTPSA(mol), 2)
            rotb    = rdMolDescriptors.CalcNumRotatableBonds(mol)
            rings   = rdMolDescriptors.CalcNumRings(mol)
            ar_r    = rdMolDescriptors.CalcNumAromaticRings(mol)
            fsp3    = round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
            qed_val = round(QED.qed(mol), 3)
            logs    = _esol_logs(mol)
            sa      = _sa_score_rdkit(mol)
            viols   = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            hac     = mol.GetNumHeavyAtoms()

            # PAINS check
            from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            catalog = FilterCatalog(params)
            pains_flag = "Alert" if catalog.HasMatch(mol) else "Clean"

            # Veber oral bioavailability
            veber_ok = "Pass" if rotb <= 10 and tpsa <= 140 else "Fail"

            # Pfizer 3/75 rule
            pfizer = "Pass" if logp <= 3 and tpsa >= 75 else "Fail"

            records.append({
                "Name":            name,
                "ΔG (kcal/mol)":  round(score, 3),
                # Physicochemical
                "MW":              mw,
                "LogP":            logp,
                "LogD7.4":         round(logp - 0.2 * (7.4 - 7.0), 2),
                "LogS (ESOL)":     logs,
                "HBD":             hbd,
                "HBA":             hba,
                "TPSA":            tpsa,
                "RotBonds":        rotb,
                "Rings":           rings,
                "ArRings":         ar_r,
                "Fsp3":            fsp3,
                "HeavyAtoms":      hac,
                # ADME
                "BBB":             _bbb_heuristic(mol),
                "HIA":             "High" if tpsa < 100 and mw < 500 else "Low",
                "Caco2":           "High" if tpsa < 120 and mw < 500 else "Low",
                # Drug-likeness
                "QED":             qed_val,
                "SA_Score":        sa,
                "LipinskiViol":    viols,
                "Lipinski":        "Pass" if viols == 0 else "Fail",
                "Veber":           veber_ok,
                "Pfizer_3_75":     pfizer,
                "PAINS":           pains_flag,
                "DrugLikeness":    _drug_likeness_label(qed_val, viols),
                # Toxicity flags
                "hERG_Risk":       _herg_risk(mol),
                "AMES_Flag":       "Unknown",  # Requires model
                "DILI_Flag":       "Unknown",  # Requires model
            })
        except Exception as e:
            records.append({"Name": name, "ΔG (kcal/mol)": round(score, 3),
                            "Error": str(e)})

    return pd.DataFrame(records)


# ── Main entry point ──────────────────────────────────────────────────────────
def run_admet_analysis(
    mols:        List[Chem.Mol],
    names:       List[str],
    scores:      List[float],
    use_api:     bool = True,
    status_text  = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Run ADMET analysis. Tries ADMETlab 3.0 API first; falls back to RDKit.
    Returns (DataFrame, source) where source is "ADMETlab3.0" or "RDKit".
    """
    from typing import Tuple

    if use_api:
        smiles_list = []
        for mol in mols:
            try:
                smiles_list.append(Chem.MolToSmiles(mol))
            except Exception:
                smiles_list.append("")

        api_df = predict_via_admetlab3(smiles_list, names, status_text)
        if api_df is not None and not api_df.empty:
            # Merge ΔG scores
            api_df.insert(1, "ΔG (kcal/mol)",
                          [round(s, 3) for s in scores[:len(api_df)]])
            return api_df, "ADMETlab 3.0"

    # Fallback to RDKit
    if status_text:
        status_text.info("ADMETlab 3.0 unavailable — using RDKit ADMET descriptors.")
    rdkit_df = predict_rdkit_admet(mols, names, scores)
    return rdkit_df, "RDKit"
