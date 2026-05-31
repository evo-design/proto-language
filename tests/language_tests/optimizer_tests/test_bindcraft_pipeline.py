"""Unit coverage for the BindCraft example pipeline."""

import inspect
from pathlib import Path
from types import SimpleNamespace

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.dssp import DSSPSecondaryStructureInput

from examples.bindcraft import run_bindcraft_full as bindcraft
from proto_language.utils.alphafold2_multimer import af2_multimer_structures
from proto_language.utils.scheduling import SCHEDULES


def _bindcraft_config() -> bindcraft.BindCraftConfig:
    return bindcraft.BindCraftConfig(target_pdb=Path("target.pdb"), target_chains=["A"], optimise_beta=False)


def _af2_config() -> bindcraft.AlphaFold2MultimerStructureConfig:
    return bindcraft._make_af2_config(
        _bindcraft_config(),
        Path("examples/germinal/pdbs/pdl1.pdb").read_text(),
        seed=0,
    )


def test_dssp_secondary_structure_percentages_uses_dssp_tool(monkeypatch) -> None:
    """BindCraft secondary-structure checks should use the DSSP tool wrapper."""
    structure = Structure(structure=str(Path("tests/dummy_data/test_structure_similarity.pdb")))
    calls: list[DSSPSecondaryStructureInput] = []

    def fake_run_dssp(inputs: DSSPSecondaryStructureInput) -> SimpleNamespace:
        calls.append(inputs)
        metrics = SimpleNamespace(helix_pct=12.5, sheet_pct=25.0, loop_pct=62.5)
        return SimpleNamespace(results=[metrics])

    monkeypatch.setattr(bindcraft, "run_dssp_secondary_structure", fake_run_dssp)

    assert bindcraft._dssp_secondary_structure_percentages(structure, "A") == {
        "helix": 12.5,
        "sheet": 25.0,
        "loop": 62.5,
    }
    assert len(calls) == 1
    assert calls[0].inputs[0].analyzed_chain_id == "A"


def test_bindcraft_filters_skip_missing_and_none_metrics() -> None:
    """BindCraft skips absent/None optional metrics instead of failing them."""
    filters = {
        "required": bindcraft.MetricRule(1.0, ">="),
        "optional_missing": bindcraft.MetricRule(1.0, ">="),
        "optional_none": bindcraft.MetricRule(1.0, ">="),
    }

    assert bindcraft._passes_filters({"required": 1.0, "optional_none": None}, filters)
    assert not bindcraft._passes_filters({"required": 0.5, "optional_none": None}, filters)


def test_af2_constraints_use_public_structure_configs() -> None:
    """BindCraft AF2 losses should use compiler-backed public structure constraints."""
    binder = bindcraft.Segment(length=8, sequence_type="protein", label="binder")
    target = bindcraft.Segment(sequence="ACDE", sequence_type="protein", label="target")
    af2_cfg = _af2_config()

    constraints = bindcraft._af2_constraints(
        binder,
        target,
        af2_cfg,
        {"plddt": 0.1, "i_con": 1.0},
    )

    assert [constraint.label for constraint in constraints] == ["af2_plddt", "af2_i_con"]
    assert [constraint.weight for constraint in constraints] == [0.1, 1.0]
    assert constraints[0].function is bindcraft.structure_plddt_constraint
    assert constraints[1].function is bindcraft.structure_interface_contact_constraint
    assert constraints[0].function_config is not constraints[1].function_config
    assert constraints[0].function_config.alphafold2_multimer_config is not af2_cfg


def test_pyrosetta_scoring_uses_beta_nov16() -> None:
    """BindCraft scores with beta_nov16; ref2015 is a different, non-comparable REU scale.

    Guards the shared PYROSETTA_SCORE_FUNCTION constant, that each PyRosetta config type accepts
    it, and that both scoring paths (validation scoring + trajectory relax) reference it.
    """
    assert bindcraft.PYROSETTA_SCORE_FUNCTION == "beta_nov16"
    for config_cls in (
        bindcraft.PyRosettaRelaxConfig,
        bindcraft.PyRosettaInterfaceAnalyzerConfig,
        bindcraft.PyRosettaEnergyConfig,
    ):
        assert config_cls(scorefxn=bindcraft.PYROSETTA_SCORE_FUNCTION).scorefxn == "beta_nov16"
    assert "PYROSETTA_SCORE_FUNCTION" in inspect.getsource(bindcraft._score_variant)
    assert "PYROSETTA_SCORE_FUNCTION" in inspect.getsource(bindcraft.run_trajectory)


def test_zero_optional_4stage_stages_builds_logit_only_program() -> None:
    """Smoke-sized hallucination programs can omit optional later stages."""
    config = _bindcraft_config()
    config.logit_steps = 1
    config.softmax_steps = 0
    config.hard_steps = 0
    config.semigreedy_steps = 0
    binder = bindcraft.Segment(length=8, sequence_type="protein", label="binder")
    target = bindcraft.Segment(sequence="ACDE", sequence_type="protein", label="target")

    program, stage_names = bindcraft._build_hallucination(
        config,
        binder,
        target,
        bindcraft.Construct([binder, target]),
        _af2_config(),
        {"plddt": 0.1},
        binder_length=8,
    )

    assert stage_names == ["logit_a"]
    assert len(program.optimizers) == 1


def test_run_bindcraft_extracts_target_sequence_with_structure_api(tmp_path, monkeypatch) -> None:
    """Target sequence extraction should use the current Structure API."""
    seen_target_sequences: list[str] = []

    def fake_run_trajectory(config, traj_idx, seed, target_pdb_text, target_seq, *args, **kwargs) -> int:
        seen_target_sequences.append(target_seq)
        return 0

    monkeypatch.setattr(bindcraft, "run_trajectory", fake_run_trajectory)
    config = bindcraft.BindCraftConfig(
        target_pdb=Path("examples/germinal/pdbs/pdl1.pdb"),
        target_chains=["A"],
        max_trajectories=1,
        output_dir=str(tmp_path),
        enable_rejection_check=False,
    )

    bindcraft.run_bindcraft(config)

    assert len(seen_target_sequences) == 1
    assert seen_target_sequences[0].startswith("AFTVTVPK")


def test_4stage_semigreedy_uses_plddt_position_weighting() -> None:
    """Default 4stage semigreedy should target low-pLDDT positions."""
    binder = bindcraft.Segment(length=8, sequence_type="protein", label="binder")
    target = bindcraft.Segment(sequence="ACDE", sequence_type="protein", label="target")
    program, stage_names = bindcraft._build_hallucination(
        _bindcraft_config(),
        binder,
        target,
        bindcraft.Construct([binder, target]),
        _af2_config(),
        {"plddt": 0.1},
        binder_length=8,
    )

    semigreedy = program.optimizers[stage_names.index("semigreedy")].generators[0]

    assert semigreedy.config.position_weighting == "plddt"
    assert semigreedy.config.clear_logits is True


def test_softmax_stage_anneals_learning_rate() -> None:
    """The softmax stage decays its learning rate, matching ColabDesign's lr ~ temperature.

    Regression: a constant lr_schedule held the effective LR at the full ``lr`` for the whole
    softmax stage; upstream decays it ~lr -> lr*1e-2 across the temperature anneal.
    """
    config = _bindcraft_config()
    config.logit_steps, config.softmax_steps, config.hard_steps, config.semigreedy_steps = 0, 10, 0, 0
    binder = bindcraft.Segment(length=8, sequence_type="protein", label="binder")
    target = bindcraft.Segment(sequence="ACDE", sequence_type="protein", label="target")

    program, stage_names = bindcraft._build_hallucination(
        config, binder, target, bindcraft.Construct([binder, target]), _af2_config(), {"plddt": 0.1}, binder_length=8
    )

    softmax = program.optimizers[stage_names.index("softmax")].config
    assert softmax.lr_schedule == "quadratic"
    assert softmax.scale_lr_by_temperature is True
    schedule = SCHEDULES[softmax.lr_schedule](softmax.temperature_start, softmax.temperature_end)
    assert schedule(softmax.num_steps, softmax.num_steps) < schedule(1, softmax.num_steps)


def test_empty_relaxed_interface_skips_mpnn_redesign(tmp_path: Path) -> None:
    """Upstream BindCraft does not redesign when the relaxed interface is empty."""
    empty_interface = SimpleNamespace(interface_contact_residues=lambda **_: {})

    accepted = bindcraft._redesign_and_validate(
        _bindcraft_config(),
        binder=SimpleNamespace(),
        construct=SimpleNamespace(),
        complex_struct=SimpleNamespace(),
        mpnn_complex_struct=empty_interface,
        binder_struct=SimpleNamespace(),
        target_struct=SimpleNamespace(),
        target_pdb_text="",
        target_seq="ACDE",
        traj_idx=0,
        run_dir=tmp_path,
        seen_sequences=set(),
    )

    assert accepted == 0


def test_redesign_perplexity_constraint_scores_binder_not_complex(tmp_path: Path, monkeypatch) -> None:
    """The MPNN-redesign perplexity constraint scores the binder against the binder backbone.

    Regression: scoring a binder-only sequence against the full target+binder complex raised a
    logits/structure length-mismatch ValueError in proteinmpnn-gradient, aborting the
    rejection-sampling redesign stage on the first batch.
    """
    complex_struct = Structure(structure=Path("examples/germinal/pdbs/pdl1.pdb").read_text())
    binder_struct = complex_struct.select_chain(bindcraft.BINDER_CHAIN)
    binder_len = len(binder_struct.get_chain_positions(bindcraft.BINDER_CHAIN))
    binder = bindcraft.Segment(length=binder_len, sequence_type="protein", label="binder")
    target = bindcraft.Segment(sequence="ACDE", sequence_type="protein", label="target")

    captured: dict[str, list] = {}

    class _CaptureOptimizer:
        def __init__(self, *, constraints, **_) -> None:
            captured["constraints"] = constraints

    class _NoopProgram:
        def __init__(self, *_, **__) -> None: ...
        def run_stage(self, *_) -> None: ...

    monkeypatch.setattr(bindcraft, "RejectionSamplingOptimizer", _CaptureOptimizer)
    monkeypatch.setattr(bindcraft, "Program", _NoopProgram)

    config = _bindcraft_config()
    config.max_mpnn_per_trajectory = 0  # never reach the (mocked) scoring loop
    # Non-empty relaxed interface so the redesign proceeds to build the constraint.
    mpnn_complex = SimpleNamespace(interface_contact_residues=lambda **_: {1: "ALA", 2: "GLY"})

    accepted = bindcraft._redesign_and_validate(
        config,
        binder=binder,
        construct=bindcraft.Construct([binder, target]),
        complex_struct=complex_struct,
        mpnn_complex_struct=mpnn_complex,
        binder_struct=binder_struct,
        target_struct=SimpleNamespace(),
        target_pdb_text="",
        target_seq="ACDE",
        traj_idx=0,
        run_dir=tmp_path,
        seen_sequences=set(),
    )

    assert accepted == 0
    (constraint,) = captured["constraints"]
    structure_input = constraint.function_config.structure_input
    assert constraint.label == "proteinmpnn_perplexity"
    # Scored against the binder-only structure, so a binder-only one-hot matches the parsed
    # structure length (the contract proteinmpnn-gradient enforces; the complex is larger).
    assert list(structure_input.chains_to_redesign.chains) == [bindcraft.BINDER_CHAIN]
    assert len(structure_input.structure.get_chain_positions(bindcraft.BINDER_CHAIN)) == binder_len
    complex_len = sum(len(complex_struct.get_chain_positions(c)) for c in ("A", bindcraft.BINDER_CHAIN))
    assert complex_len > binder_len


def test_af2_multimer_structures_selects_chain_b_for_de_novo() -> None:
    """De-novo (binder_chain=None) extracts the binder from output chain 'B'; the target stays 'A'."""
    complex_struct = Structure(structure=Path("examples/germinal/pdbs/pdl1.pdb").read_text())
    config = bindcraft.AlphaFold2MultimerStructureConfig(binder_chain=None)
    binder_struct, target_struct = af2_multimer_structures(complex_struct, config, n_inputs=2)
    assert len(binder_struct.get_chain_positions("B")) == 118
    assert len(target_struct.get_chain_positions("A")) == 115
