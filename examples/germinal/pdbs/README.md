# Germinal PDB fixtures

Copied from [`SantiagoMille/germinal/pdbs`](https://github.com/SantiagoMille/germinal/tree/main/pdbs) (Apache 2.0 — redistribution permitted with attribution).

| File | Chain A | Chain B |
|---|---|---|
| `pdl1.pdb` | PD-L1 target (115 residues) | bound nanobody (118 residues) |
| `il3.pdb` | IL-3 target (112 residues) | — |
| `insulin.pdb` | insulin A chain (21 residues) | insulin B chain (30 residues) |
| `nb.pdb` | VHH scaffold (131 residues) | — |
| `scfv.pdb` | scFv scaffold (242 residues) | — |

Germinal target hotspots (from `configs/target/*.yaml`):

- `pdl1.pdb` — `A37, A39, A41, A96, A98`
