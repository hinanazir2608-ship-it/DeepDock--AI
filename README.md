# 🧬 DeepDock-AI: Automated Virtual Screening & Molecular Docking Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepdock--ai.streamlit.app)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**DeepDock-AI** is an automated computational platform designed for small-molecule inhibitor screening, structure-based drug design, and molecular docking workflows. Built specifically for researchers and computational biochemists to streamline structure preparation and binding affinity evaluation.

---

## ✨ Features

- **Automated Virtual Screening:** Screen datasets of small-molecule ligands against target proteins seamlessly.
- **Molecular Docking Engine:** Integration with **AutoDock Vina** for fast binding energy calculation ($\text{kcal/mol}$).
- **Cheminformatics Pipelines:** Ligand preparation, SMILES parsing, and format conversion using **RDKit**, **Meeko**, and **OpenBabel**.
- **Interactive 3D Visualization:** In-browser visualization of protein-ligand interactions via **py3Dmol** and **Streamlit Molstar**.

---

## 🛠️ Tech Stack

- **UI Framework:** Streamlit
- **Core Engine:** AutoDock Vina
- **Cheminformatics Libraries:** RDKit, OpenBabel, Meeko, Gemmi
- **Language:** Python 3.10

---

## 🔬 Usage Workflow

1. **Upload Target Protein:** Input prepared protein structure (`PDB` or `PDBQT` format).
2. **Input Ligands:** Provide SMILES strings or upload small-molecule structure files (`SDF` / `PDBQT`).
3. **Configure Search Space:** Set binding site center coordinates ($X, Y, Z$) and box size.
4. **Run Docking:** Execute AutoDock Vina to calculate binding energy poses.
5. **Analyze Results:** View pose tables, download docked configurations, and inspect 3D interactions interactively.

---

## 🚀 Cloud Deployment Note

This project is optimized for deployment on **Streamlit Community Cloud**. It includes system-level Boost dependencies specified in `packages.txt` and pins Python to version `3.10` via `.python-version` to ensure seamless C++ binary binding for AutoDock Vina.

### Local Installation

```bash
# Clone repository
git clone [https://github.com/hinanazir2608-ship-it/DeepDock--AI.git](https://github.com/hinanazir2608-ship-it/DeepDock--AI.git)
cd DeepDock--AI

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
