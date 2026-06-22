# HTH–DNA docking templates

The 20 natural helix-turn-helix (HTH) protein–DNA crystal structures used by the
ProtoRepressor pipeline (`examples/scripts/protorepressor.py`) as **docking templates**
for template-guided superposition (see `examples/scripts/protorepressor_templating.py`).

PDB sources: 1QPI, 1R8D, 2KEI, 2OR1, 2VZ4, 2XRO, 2ZHG, 3BDN, 3ZQL, 4EGY, 4EGZ, 4L62,
4PXI, 4WLS, 5D8C, 5YEJ, 6JGW, 7TEA, 7TEC, 8SVD.

These are **not** per-operator start models. Each file is one bound protein–DNA complex
(≤2 protein protomers + its DNA duplex), reduced from the full crystal asymmetric unit
with `protorepressor_templating.extract_bound_complex` (waters, ligands, and extra
asymmetric-unit copies removed). At design time, Stage 1 survivors are positioned on the
idealized B-form operator DNA by superposing each candidate's recognition helix onto a
template — these crystals are the only structural data the pipeline ships, and start
models are generated in-pipeline from Stage-1 output.
