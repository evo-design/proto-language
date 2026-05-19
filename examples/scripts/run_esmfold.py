"""Predict a single protein structure with ESMFold and write it as CIF."""

import sys

from proto_tools import ESMFoldConfig, ESMFoldInput, run_esmfold

if __name__ == "__main__":
    inputs = ESMFoldInput(complexes=[sys.argv[1]])
    config = ESMFoldConfig()
    output = run_esmfold(inputs, config)

    with open("design.cif", "w") as f:
        f.write(output.structures[0].structure_cif)
