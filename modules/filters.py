import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def apply_rdkit_filters(file_path, filter_type="Lipinski"):
    mols = []
    ext = os.path.splitext(file_path)[-1].lower()

    # Load SDF / MOL files safely without dropping 3D structures
    if ext in ['.sdf', '.mol']:
        supplier = Chem.SDMolSupplier(file_path, sanitize=False, removeHs=False)
        for m in supplier:
            if m is not None:
                try:
                    Chem.SanitizeMol(m)
                except:
                    pass
                mols.append(m)

    elif ext in ['.csv', '.smi', '.txt']:
        df = pd.read_csv(file_path)
        smiles_col = next((col for col in df.columns if 'smiles' in col.lower() or 'smi' in col.lower()), None)
        if smiles_col:
            for s in df[smiles_col]:
                if pd.notna(s):
                    m = Chem.MolFromSmiles(str(s))
                    if m is not None:
                        mols.append(m)

    filtered_mols = []
    filtered_names = []
    filtered_cids = []
    table_data = []

    for idx, mol in enumerate(mols):
        # Safe Name Handling
        mol_name = ""
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            mol_name = mol.GetProp("_Name").strip()
        elif mol.HasProp("TITLE") and mol.GetProp("TITLE").strip():
            mol_name = mol.GetProp("TITLE").strip()
        else:
            mol_name = f"Ligand_{idx + 1}"

        mol.SetProp("_Name", mol_name)

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hdonors = Lipinski.NumHDonors(mol)
        hacceptors = Lipinski.NumHAcceptors(mol)

        if filter_type == "Lipinski":
            if mw <= 500 and logp <= 5 and hdonors <= 5 and hacceptors <= 10:
                filtered_mols.append(mol)
                filtered_names.append(mol_name)
                filtered_cids.append(idx + 1)
                table_data.append({
                    "CID": idx + 1,
                    "Name": mol_name,
                    "MW": round(mw, 2),
                    "LogP": round(logp, 2),
                    "HBD": hdonors,
                    "HBA": hacceptors
                })
        else:
            filtered_mols.append(mol)
            filtered_names.append(mol_name)
            filtered_cids.append(idx + 1)
            table_data.append({
                "CID": idx + 1,
                "Name": mol_name,
                "MW": round(mw, 2),
                "LogP": round(logp, 2),
                "HBD": hdonors,
                "HBA": hacceptors
            })

    filter_df = pd.DataFrame(table_data)
    return filtered_mols, filtered_names, filtered_cids, filter_df