# Component Planning Example

A worked Phase 2 `component_planning.md` for a small, concrete task — de novo
PD-L1 mini-binder design — written to match the actual division of labor in this
codebase. Use it as a shape reference for what a component plan should contain:
segments, generators, constraints, optimizer stages, a validation panel, and
deep-dives. For the compressed rules this example follows, see
`notes/planning-quick-reference.md`.

**Implementation premise.** RFDiffusion3 is available in the workspace as a
proto-tools structure-design tool; ProteinMPNN is available as a native
proto-language generator (`proteinmpnn`). So this plan uses a hybrid pipeline:

- Stage 1: tool-level backbone design with `proto-tools:rfdiffusion3-design`.
- Stage 2 onward: native proto-language generation, optimization, and validation.

## 1. Segments and Constructs

| Object | Type | Length | Fixed / Mutable | Notes |
|---|---|---|---|---|
| `target_segment` | protein (aa) | 117 | fully fixed | PD-L1 IgV domain residues 18–134 from 5O45 chain A. Loaded from staged PDB; sequence and structure both used. |
| `binder_segment` | protein (aa) | 60–80 | fully mutable | The candidate mini-binder. Stage 1 produces backbone coordinates for this segment; Stage 2 writes sequence proposals for it. |
| `complex_construct` | protein–protein complex | — | composed | Joins `target_segment` (fixed) with `binder_segment` (mutable) for binding evaluation. Consumed by structure-based binding constraints. |
| `binder_backbone_pool` | staged structural pool | 60–80 aa per backbone | fixed after Stage 1 | Set of candidate binder backbones produced by `rfdiffusion3-design` before sequence design begins. |

No multi-chain / oligomeric configuration beyond the intended 1:1 PD-L1:binder
complex.

## 2. Generator Plan

### Stage 1 — RFDiffusion3 Backbone Design (`proto-tools:rfdiffusion3-design`)

- **Target segment:** generates backbone coordinates for `binder_segment`
  conditioned on a hotspot patch on `target_segment`.
- **Conditioning inputs:** `target_segment` structure from 5O45, plus
  hotspot/interface residues on the PD-1 contact face of PD-L1 derived from 4ZQK
  and summarized in deep-dive §A.
- **Relevant config fields:** `length`, `input_structure`, `contig`,
  `select_hotspots`, optional partial-diffusion controls, and global sampling
  controls in `RFdiffusion3Config`.
- **Role in pipeline:** produces candidate binder backbones only, not final
  amino acid sequences.
- **Reason for choice:** this is the actual structure-design component present in
  the workspace for de novo backbone generation. It is the correct upstream step
  for binder design against a fixed target surface.

### Stage 2 — ProteinMPNN (`proteinmpnn`)

- **Target segment:** writes sequence proposals for `binder_segment` given
  Stage 1 backbones.
- **Conditioning inputs:** Stage 1 backbone outputs passed as `structure_inputs`.
- **Relevant config fields:** `temperature=0.1`, `excluded_amino_acids=["C"]`,
  optional chain-specific design settings if needed.
- **Role in pipeline:** converts Stage 1 candidate backbones into protein
  sequences predicted to realize those backbones.
- **Reason for choice:** this is the native inverse-folding generator available
  in proto-language and is the natural downstream partner for RFDiffusion-
  designed backbones.

### Generator Integration Note

This is not a fully registry-native two-generator pipeline inside
`proto_language.language.generator`. Instead:

- Stage 1 is a proto-tools structure-design step.
- Stage 2 is a native proto-language generator step.

That is sufficient for the task, but it should be documented explicitly so
readers do not infer that `rfdiffusion3` is already wrapped as a registered
language generator.

## 3. Constraint Plan

Constraints below use actual available
proto-language registry entries where possible, with explicit notes where
postprocessing is external.

| # | Name | Role | Direction | Threshold | Stage | Inputs | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `protein-length` | cheap filter | within | 60 ≤ L ≤ 80 | Stage 2 onward | `binder_segment` | Real registry constraint for protein length. |
| 2 | `protein-diversity` | cheap quality filter | maximize | e.g. minimum diversity ≥ 0.6 | Stage 2 onward | `binder_segment` | Real registry constraint on amino-acid diversity. |
| 3 | `balanced-aa` | cheap quality filter | maximize | no severe AA imbalance | Stage 2 onward | `binder_segment` | Guards against pathological residue compositions. |
| 4 | `overall-protein-quality` | cheap composite filter | maximize | pass internal sub-thresholds | Stage 2 onward | `binder_segment` | Catches repetitiveness / low complexity / poor composition. |
| 5 | generator config `excluded_amino_acids=["C"]` | generation-time filter | exclude | no cysteine | Stage 2 | `binder_segment` | There is no separate built-in no-cysteine constraint; enforce in generator config and audit outputs. |
| 6 | `structure-plddt` with `structure_tool="esmfold"` | medium-cost monomer filter | maximize confidence | equivalent to mean pLDDT ≥ 75–80 | post-Stage-2 | `binder_segment` | Fast monomer fold screen. Constraint returns normalized energy, so threshold is applied on transformed score. |
| 7 | `structure-plddt` with `structure_tool="boltz2"` or `alphafold3` | monomer validation | maximize confidence | equivalent to mean pLDDT ≥ 80 | post-fold-screen | `binder_segment` | Stronger monomer confirmation using a second predictor family. |
| 8 | `structure-iptm` with `structure_tool="boltz2"` or `alphafold3` | binding validation | maximize interface confidence | high ipTM | post-monomer | `complex_construct` | Primary binding-confidence signal for the PD-L1:binder complex. |
| 9 | `structure-pae` with `structure_tool="boltz2"` or `alphafold3` | binding validation | minimize | equivalent to interface / complex pAE ≤ 10 Å | post-monomer | `complex_construct` | Real pAE-based complex-confidence signal. |
| 10 | `mmseqs-gene-similarity` | novelty filter | minimize similarity | best-hit identity < 30% | post-binding | `binder_segment` | Real registry novelty filter against a staged UniRef50 MMseqs DB. |
| 11 | external pairwise identity selection | diversity guard | — | pairwise identity ≤ 60% among finals | final selection | full surviving pool | Not currently a native registry constraint; perform in postprocessing. |

### Threshold Translation

Some biological requirements are specified in raw model metrics, while the
actual constraints return normalized energies where lower is better. For this
plan:

- `structure-plddt` returns approximately `1 − normalized_pLDDT`
- `structure-iptm` returns approximately `1 − ipTM`
- `structure-pae` returns approximately `avg_pAE / 31.75`

So the biological statement `mean pLDDT ≥ 80` maps approximately to
`structure-plddt ≤ 0.20`, and `pAE ≤ 10 Å` maps approximately to
`structure-pae ≤ 10 / 31.75 ≈ 0.315`. These translations should be made explicit
in implementation.

### Oracle Agreement

Requirement (a), monomer fold stability, uses two structure oracles:

- `structure-plddt` with `esmfold`
- `structure-plddt` with `boltz2` or `alphafold3`

Requirement (b), binding, uses complex structure confidence from
`structure-iptm` and `structure-pae`. This is stronger than a single scalar
alone, but still ultimately depends on structure-prediction-based binding
surrogates rather than experimental affinity data.

Total cost is still high enough that the pipeline should be described as a
substantial compute job rather than a casual screening run.

## 4. Optimizer Plan

Composed pipeline with an external backbone-design stage followed by native
inverse folding and validation.

- **Stage 1 — Backbone generation.** Run `proto-tools:rfdiffusion3-design`.
  Generate a broad pool of hotspot-conditioned binder backbones against PD-L1.
  Output: `binder_backbone_pool`. This is an external tool stage rather than a
  native proto-language optimizer stage.
- **Stage 2 — Sequence generation on fixed backbones.** Use `TopKOptimizer` over
  `proteinmpnn`. Inputs: `binder_backbone_pool`. Purpose: sample many sequence
  candidates for the designed backbones and retain the best early-quality
  sequences. Cheap constraints active here: `protein-length`,
  `protein-diversity`, `balanced-aa`, `overall-protein-quality`.
- **Stage 3 — Optional iterative self-consistency refinement.** Use
  `CyclingOptimizer` with `proteinmpnn` if desired (pipeline `protein-hunter`,
  conditioning parameter `structure_inputs`). Purpose: iteratively refine
  sequence/backbone self-consistency for surviving binders. This stage is
  optional; it improves monomer self-consistency but is not itself
  target-conditioned binder optimization.
- **Stage 4 — Monomer validation.** Run a follow-on screening stage with
  `structure-plddt` using `esmfold` and `structure-plddt` using `boltz2` or
  `alphafold3`. Purpose: remove backbones/sequences that do not support a
  confident monomeric fold.
- **Stage 5 — Complex binding validation.** Run complex scoring on
  (`target_segment`, `binder_segment`) using `structure-iptm` and
  `structure-pae`. Purpose: identify candidates predicted both to fold and to
  form a plausible complex with PD-L1.
- **Stage 6 — Novelty filter.** Apply `mmseqs-gene-similarity` with the UniRef50
  MMseqs database and novelty cutoff < 30% identity.
- **Stage 7 — Final diversity selection.** External postprocessing stage: sort
  surviving candidates by binding score; greedily retain candidates whose
  pairwise identity to accepted candidates is ≤ 60%; stop at 50 candidates or
  when the pool is exhausted.

### Stopping Criterion

The pipeline succeeds when Stage 7 yields 50 candidates. If fewer than 50
survive:

- return all survivors,
- flag the run as underfilled,
- report the bottleneck stage if identifiable.

## 5. Optimization Stages

- `rfdiffusion3-design` generates hotspot-conditioned binder backbones against
  PD-L1.
- `TopKOptimizer` with `proteinmpnn` generates sequences on those backbones.
- Optional `CyclingOptimizer` improves sequence/structure self-consistency for
  the binder monomer.
- Monomer fold screening is applied with `structure-plddt`.
- Complex binding screening is applied with `structure-iptm` and `structure-pae`.
- Novelty filtering is applied with `mmseqs-gene-similarity`.
- Final-set diversity is enforced by external pairwise sequence identity pruning.

No stage after Stage 2 introduces a new generator prior. Stages 4–7 are
validation and selection.

## 6. Validation Panel

The final validation panel on the 50 returned candidates should emit a
per-candidate report with:

| Quantity | Tool / Constraint | Threshold | Comparison Set | Secondary Oracle |
|---|---|---|---|---|
| Monomer fold quality | `structure-plddt` with `esmfold` | translated from raw pLDDT target | — | `structure-plddt` with `boltz2` or `alphafold3` |
| Monomer fold confirmation | `structure-plddt` with `boltz2` or `alphafold3` | translated from raw pLDDT target | — | `esmfold` |
| Binding interface quality | `structure-iptm` | high-confidence threshold | — | paired with `structure-pae` |
| Binding alignment error | `structure-pae` | ≤ 0.315 normalized (raw pAE ≤ 10 Å) | — | paired with `structure-iptm` |
| Novelty | `mmseqs-gene-similarity` | < 30% identity | UniRef50 | none |
| Diversity within final set | external pairwise alignment | ≤ 60% pairwise identity | the 50 final candidates | — |

Weakest legs of the validation panel:

- final pairwise diversity is external rather than native;
- binding remains prediction-based rather than experimentally calibrated.

That limitation should be documented in the final report.

## 7. Reference Scripts and Components

Closest actual references in this codebase:

- `protein_hunter.py`
- `proto_language.language.generator.proteinmpnn_generator`
- `proto_language.language.optimizer.topk_optimizer`
- `proto_language.language.optimizer.cycling_optimizer`
- `proto_language.language.constraint.protein_structure.structure_confidence_constraint`
- `proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint`

What is reused:

- native `proteinmpnn` inverse folding
- native topk optimizer for broad sampling
- optional native cycling refinement
- native structure-confidence constraints
- native MMseqs novelty filter

What is not yet native:

- `rfdiffusion3` as a registered proto-language generator
- a first-class binder-design optimizer abstraction
- native final-set pairwise diversity selection

## 8. Deep-Dive Reports

### §A — Generator Deep-Dive: De Novo Binder Design Against PD-L1

- **Question:** What is the best available generator plan in this codebase for de
  novo mini-binder design against PD-L1 with hotspot conditioning on the PD-1
  contact face?
- **Sources read:** workspace tool inventory, rfdiffusion3 tool docs, proteinmpnn
  generator docs, `protein_hunter.py`, and relevant structure-design tool
  references.
- **Recommended primary:** a two-stage RFDiffusion3 → ProteinMPNN pipeline.
  - Stage 1: `proto-tools:rfdiffusion3-design` for target-conditioned backbone
    generation.
  - Stage 2: `proteinmpnn` for inverse-folding sequence generation on those
    backbones.
- **Why this is the best fit:** it matches the expected de novo binder workflow
  and uses components that actually exist in the workspace.
- **Implementation caveat:** only Stage 2 is currently a native generator in the
  proto-language registry. Stage 1 is available as a tool but not yet as a
  registered language generator.
- **Hotspot residue selection on PD-L1:** use the PD-1 contact face derived from
  4ZQK, emphasizing interface residues such as Y56, E58, R113, M115, and Y123 as
  the core hotspot-conditioning set. These lie on the canonical PD-1-binding face
  of the PD-L1 IgV domain.
- **Cysteine exclusion:** set `excluded_amino_acids=["C"]` in
  `ProteinMPNNGeneratorConfig` to reduce free-cysteine liabilities in monomeric
  designs.

### §B — Tool Deep-Dive: Binding Validation

- **Question:** What is the best in-repo structure-based proxy for binding
  quality?
- **Recommended primary:** joint use of `structure-iptm` and `structure-pae` on
  the predicted `target_segment` + `binder_segment` complex, using `boltz2` or
  `alphafold3`.
- **Reason:** these are the strongest native structure-based interface signals
  exposed by the current constraint layer.
- **Ruled-out phrasing:** referring to af2_multimer as though it were the native
  constraint endpoint here. In this repo, the actual language-level abstractions
  are the structure-confidence constraints with configurable structure tools.

### §C — Tool Deep-Dive: Novelty Against UniRef50

- **Question:** Which native constraint should implement the < 30% identity
  novelty screen?
- **Recommended primary:** `mmseqs-gene-similarity`.
- **Configuration pattern:** `min_similarity=0`, `max_similarity=30`,
  `mmseqs_db=<staged_uniref50_mmseqs_db>`.
- **Reason:** this is the actual registry-native similarity constraint and is the
  correct match for a UniRef50 novelty screen.
- **Limitation:** it is named generally as a gene/protein similarity constraint
  rather than as a purpose-built novelty selector, so the planning document
  should spell out that it is being used specifically as a novelty filter.
