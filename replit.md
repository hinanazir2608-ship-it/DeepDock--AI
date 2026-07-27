# DeepDock-AI

GPU-Accelerated Virtual Screening Platform — predicts binding affinity (ΔG) for ligand libraries against a target protein using a PyTorch GNN with full ADMET profiling and export.

## Run & Operate

- `streamlit run app.py --server.port 5000` — start the DeepDock-AI web app
- Workflow: **DeepDock-AI** (auto-configured, port 5000)

## Stack

- **UI:** Streamlit (app.py)
- **Deep Learning:** PyTorch — 3-layer GCN + Attention Pooling GNN
- **GPU:** CUDA FP16 AMP autocast (Kaggle T4 / P100 optimised)
- **Cheminformatics:** RDKit (SDF, PDB, ADMET, PAINS, QED, ESOL)
- **Parallel filtering:** Python `multiprocessing`
- **Export:** CSV, Parquet (pyarrow), SDF (SDWriter), DOCX (python-docx)

## Where things live

- `app.py` — main Streamlit UI (all 5 pipeline phases)
- `modules/filters.py` — Module 1: parallel Lipinski + Veber + PAINS pre-filter
- `modules/graph_pipeline.py` — Module 2: mol→graph tensor dataset + DataLoader
- `modules/model.py` — Module 3: DeepDockGNN + `@st.cache_resource` loader
- `modules/inference.py` — Module 4: batch GPU inference + CUDA OOM recovery
- `modules/admet_export.py` — Module 5: ADMET profiling, SDF export, DOCX report
- `modules/preprocessing.py` — SDF/PDB/conf.txt parsing utilities
- `.streamlit/config.toml` — Streamlit server config (port 5000, dark theme)

## Pipeline Flow

1. Upload `Ligands.sdf` + `protein.pdb` (+ optional `conf.txt`)
2. Parallel 2D ADMET pre-filter (Lipinski RO5 + Veber + PAINS) — bypass toggle for ≤20 ligands
3. Build molecular graph tensors (`LigandGraphDataset`)
4. Load GNN model onto CUDA once via `@st.cache_resource`
5. Batch GPU inference with `torch.no_grad()` + FP16 `autocast`; OOM auto-halves batch size
6. Rank by ΔG → ADMET profiling (QED, TPSA, LogP, MW, ESOL LogS, BBB)
7. Export: CSV / Parquet / consolidated SDF (`top_hits_docked.sdf`) / DOCX report

## Architecture decisions

- **Pure PyTorch GNN (no PyG):** degree-normalised adjacency (`D^{-1/2} A D^{-1/2}`) for message passing avoids torch-geometric dependency issues in Replit/Kaggle environments.
- **`@st.cache_resource` for model:** model is loaded to GPU once per session, never reloaded on re-runs.
- **Pickle-safe multiprocessing:** ligands are serialised as SMILES strings (not RDKit Mol objects) before IPC to avoid cross-process pickling failures.
- **OOM recovery:** `RuntimeError: out of memory` is caught per-batch; remaining work is re-batched at half the previous batch size and retried automatically.
- **ESOL LogS:** aqueous solubility estimated via Delaney 2004 model (no external service needed).

## User preferences

- Target Kaggle Dual T4 / P100 GPU environment
- FP16 AMP enabled when CUDA available; graceful CPU fallback
- Module descriptions provided incrementally — extend each module as new specs arrive

## Gotchas

- Run codegen is not applicable (Python stack, no OpenAPI)
- `num_workers=0` on DataLoader required when called from Streamlit (no forked subprocesses inside the main thread)
- `SDMolSupplier` used via `ForwardSDMolSupplier` to support byte-stream input without temp files
