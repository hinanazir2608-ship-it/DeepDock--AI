    """
    total = len(filtered_mols)
    if total == 0:
        return []

    target_path = get_file_path(target_pdb_file)
    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError("Target protein PDB was not found.")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="deepdock_gnina_")
    os.makedirs(output_dir, exist_ok=True)

    if not grid_center or len(grid_center) != 3:
        raise ValueError("grid_center must contain exactly three coordinates.")
    if not grid_size or len(grid_size) != 3:
        raise ValueError("grid_size must contain exactly three dimensions.")

    cx = tuple(float(x) for x in grid_center)
    size = tuple(float(x) for x in grid_size)

    receptor_clean = os.path.join(output_dir, "receptor_clean.pdb")
    info = clean_receptor_pdb(target_path, receptor_clean)

    ok, error, n_in_box = validate_grid(
        cx,
        size,
        receptor_pdb=receptor_clean,
    )
    if not ok:
        raise ValueError(error)

    gnina_path = get_gnina_path()
    use_gpu, hardware = select_hardware(force_cpu)
    gnina_version = get_gnina_version(gnina_path)

    print(f"GNINA: {gnina_path}")
    print(f"GNINA version: {gnina_version}")
    print(f"Hardware: {hardware}")
    print(f"GNINA seed: {GNINA_SEED}")
    print(f"CPU threads: {GNINA_CPU}")
    print(f"Exhaustiveness: {GNINA_EXHAUSTIVENESS}")
    print(f"Num modes: {GNINA_NUM_MODES}")
    print(f"CNN scoring: rescore")
    print(f"Grid center: {cx}")
    print(f"Grid size: {size}")
    print(f"Receptor atoms in box: {n_in_box}")
    print(f"Receptor atoms kept: {info['atoms']}")

    results = []
    for idx, mol in enumerate(filtered_mols):
        if progress is not None:
            progress(
                (idx + 1) / total,
                desc=f"GNINA docking [{idx + 1}/{total}]",
            )

        result = dock_single_ligand(
            mol=mol,
            idx=idx,
            receptor_pdb=receptor_clean,
            grid_center=cx,
            grid_size=size,
            output_dir=output_dir,
            gnina_path=gnina_path,
            use_gpu=use_gpu,
        )
        results.append(result)

        if result["Docking Status"] == "Success":
            print(
                f"[{idx + 1}/{total}] {result['Name']} | "
                f"Affinity={result['Affinity (kcal/mol)']} | "
                f"CNN Score={result['CNN Score']} | "
                f"CNN Affinity={result['CNN Affinity']}"
            )
        else:
            print(f"[{idx + 1}/{total}] {result['Name']} | {result['Docking Status']}")

    return results


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

# app_gradio.py uses these names directly; keeping them here makes the module
# easy to import without changing the Gradio pipeline.
parse_scores = parse_gnina_scores
run_docking = run_gnina_docking


if __name__ == "__main__":
    print("DeepDock-AI GNINA docking module")
    try:
        path = get_gnina_path()
        use_gpu, hardware = select_hardware()
        print(f"GNINA: {path}")
        print(f"Version: {get_gnina_version(path)}")
        print(f"Hardware selected: {hardware}")
    except Exception as exc:
        print(f"GNINA detection failed: {exc}")

