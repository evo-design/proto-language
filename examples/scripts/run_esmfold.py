from proto_language.bio_tools.tools.structure_prediction import run_esmfold, ESMFoldConfig


if __name__ == "__main__":
    config = ESMFoldConfig(sequences=sys.argv[1])
    esmfold_output = run_esmfold(config)

    # Save structure in CIF format
    with open("design.cif", "w") as f:
        f.write(esmfold_output.structure_cif)
