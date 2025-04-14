# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import StringIO
from typing import Union, List

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile


def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    return PDBFile.read(pdb_path).get_structure(model=1)


def get_atomarray_in_residue_range(atoms: AtomArray, start: int, end: int) -> AtomArray:
    return atoms[np.logical_and(atoms.res_id >= start, atoms.res_id < end)]


def read_fasta_file(fasta_path: str) -> List[str]:
    with open(fasta_path, 'r') as file:
        return [line.strip() for line in file if not line.startswith('>')]
