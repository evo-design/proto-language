"""Tests for cancer-circuit helper constraints."""

from types import SimpleNamespace

import numpy as np
import pytest

from proto_language.constraint.protein_structure.pyrosetta_interface_constraint import (
    PyRosettaInterfaceConfig,
    pyrosetta_interface_constraint,
)
from proto_language.constraint.rna_splicing.alphagenome_splice_junction import (
    AlphaGenomeSpliceJunctionConfig,
    alphagenome_splice_junction_constraint,
)
from proto_language.constraint.rna_splicing.splice_transformer_intron_boundary import (
    SpliceTransformerIntronBoundaryConfig,
    splice_transformer_intron_boundary,
)
from proto_language.constraint.rna_splicing.splice_transformer_specificity import (
    SpliceTransformerSpecificityConfig,
    splice_transformer_specificity,
)
from proto_language.constraint.rna_splicing.splice_transformer_target import (
    splice_target_window_start,
)
from proto_language.constraint.sequence_annotation.alphagenome_interval_track_constraint import (
    AlphaGenomeIntervalTrackConfig,
    alphagenome_interval_track_constraint,
)
from proto_language.constraint.sequence_annotation.mirna_specificity_constraint import (
    MiRNASpecificityConfig,
    mirna_specificity_constraint,
)
from proto_language.constraint.sequence_annotation.puffin_promoter_activity_constraint import (
    PuffinPromoterActivityConfig,
    puffin_promoter_activity_constraint,
)
from proto_language.core import Sequence


def test_alphagenome_interval_track_uses_flanking_context(monkeypatch):
    config = AlphaGenomeIntervalTrackConfig(
        intervals=[(0, 10)],
        left_context="A" * 8192,
        right_context="T" * 8192,
        ontology_terms=["EFO:0001086"],
        requested_output="CAGE",
        maximize_inflection_value=1.0,
    )
    captured = {}

    def fake_run_alphagenome_predict_sequences(tool_input, tool_config):
        sequence = tool_input.sequences[0]
        captured["sequence"] = sequence
        captured["requested_outputs"] = tool_config.requested_outputs
        values = np.zeros((len(sequence), 1), dtype=float)
        target_offset = 8187
        values[target_offset : target_offset + 10, 0] = 2.0
        result = {"predictions": {"CAGE": {"values": values.tolist()}}}
        return SimpleNamespace(results=[SimpleNamespace(result=result)])

    monkeypatch.setitem(
        alphagenome_interval_track_constraint.__globals__,
        "run_alphagenome_predict_sequences",
        fake_run_alphagenome_predict_sequences,
    )

    (output,) = alphagenome_interval_track_constraint([(Sequence("C" * 10, "dna"),)], config)

    assert len(captured["sequence"]) == 16384
    assert captured["sequence"][8187:8197] == "C" * 10
    assert output.metadata["target_offset"] == 8187
    assert output.metadata["scored_intervals"] == [[8187, 8197]]
    assert output.metadata["interval_mean_signal"] == 2.0
    assert output.score < 0.5


def test_alphagenome_interval_contrastive_maximizes_margin(monkeypatch):
    config = AlphaGenomeIntervalTrackConfig(
        intervals=[(0, 10)],
        left_context="A" * 8192,
        right_context="T" * 8192,
        ontology_terms=["EFO:TARGET"],
        contrastive_ontology_terms=["EFO:HEALTHY"],
        requested_output="CAGE",
        margin_inflection_value=0.0,
        margin_sigmoid_scale=1.0,
    )

    def fake_run(tool_input, tool_config):
        sequence = tool_input.sequences[0]
        # Strong target signal, weak healthy signal -> positive margin.
        signal = 2.0 if "TARGET" in tool_config.ontology_terms[0] else 0.5
        values = np.zeros((len(sequence), 1), dtype=float)
        values[8187:8197, 0] = signal
        result = {"predictions": {"CAGE": {"values": values.tolist()}}}
        return SimpleNamespace(results=[SimpleNamespace(result=result)])

    monkeypatch.setitem(
        alphagenome_interval_track_constraint.__globals__, "run_alphagenome_predict_sequences", fake_run
    )
    (output,) = alphagenome_interval_track_constraint([(Sequence("C" * 10, "dna"),)], config)
    assert output.metadata["interval_mean_signal"] == pytest.approx(2.0)
    assert output.metadata["contrastive_mean_signal"] == pytest.approx(0.5)
    assert output.metadata["contrastive_margin"] == pytest.approx(1.5)
    # score = 1 - sigmoid(1.5) ~= 0.18 -> a positive margin is rewarded with a low penalty.
    assert output.score < 0.25


def test_puffin_promoter_activity_scores_activity_and_sharpness(monkeypatch):
    config = PuffinPromoterActivityConfig(
        left_context="A" * 325,
        right_context="T" * 325,
        track_names=["ENCODE_CAGE+"],
        activity_threshold=4.0,
        sharpness_threshold=0.5,
    )
    captured = {}

    def fake_run_puffin_prediction(tool_input, tool_config):
        captured["sequences"] = tool_input.sequences
        predictions = np.zeros((4, 10), dtype=float)
        predictions[:, 1] = [0.0, 1.0, 4.0, 3.0]
        return SimpleNamespace(results=[SimpleNamespace(predictions=predictions.tolist())])

    monkeypatch.setitem(
        puffin_promoter_activity_constraint.__globals__,
        "run_puffin_prediction",
        fake_run_puffin_prediction,
    )

    (output,) = puffin_promoter_activity_constraint([(Sequence("ACGT", "dna"),)], config)

    assert len(captured["sequences"][0]) == 654
    assert output.score < 0.5
    assert output.metadata["puffin_activity"] == 2.0
    assert output.metadata["puffin_tss_sharpness"] == 0.5


def test_mirna_specificity_minimize_penalizes_predicted_sites(monkeypatch):
    config = MiRNASpecificityConfig(
        mirna_queries=["UGAGGUAGUAGGUUGUAUAGUU"],
        mirna_ids=["miR-test"],
        direction="minimize",
        repression_threshold=2.0,
        site_score_reference=100.0,
        energy_reference=20.0,
    )

    def fake_run_miranda_scan(tool_input, tool_config):
        site = SimpleNamespace(
            mirna_id="miR-test",
            score=100.0,
            energy=-20.0,
            target_start=1,
            target_end=21,
        )
        return SimpleNamespace(results=[SimpleNamespace(target_sites=[site], num_sites=1)])

    monkeypatch.setitem(mirna_specificity_constraint.__globals__, "run_miranda_scan", fake_run_miranda_scan)

    (output,) = mirna_specificity_constraint([(Sequence("ACGTACGT", "dna"),)], config)

    assert output.score == 0.5
    assert output.metadata["mirna_num_sites"] == 1
    assert output.metadata["mirna_sites"][0]["strength"] == 1.0


def test_mirna_specificity_applies_expression_weights(monkeypatch):
    config = MiRNASpecificityConfig(
        mirna_queries=["UGAGGUAGUAGGUUGUAUAGUU", "UAGCUUAUCAGACUGAUGUUGA"],
        mirna_ids=["miR-healthy", "miR-target"],
        mirna_weights=[2.0, 0.5],
        direction="maximize",
        repression_threshold=5.0,
        site_score_reference=100.0,
        energy_reference=20.0,
    )

    def fake_run_miranda_scan(tool_input, tool_config):
        sites = [
            SimpleNamespace(mirna_id="miR-healthy", score=100.0, energy=-20.0, target_start=1, target_end=21),
            SimpleNamespace(mirna_id="miR-target", score=100.0, energy=-20.0, target_start=30, target_end=50),
        ]
        return SimpleNamespace(results=[SimpleNamespace(target_sites=sites, num_sites=2)])

    monkeypatch.setitem(mirna_specificity_constraint.__globals__, "run_miranda_scan", fake_run_miranda_scan)

    (output,) = mirna_specificity_constraint([(Sequence("ACGTACGT", "dna"),)], config)

    assert output.metadata["mirna_repression_score"] == 2.5
    assert output.score == 0.5
    assert [site["weight"] for site in output.metadata["mirna_sites"]] == [2.0, 0.5]
    assert [site["strength"] for site in output.metadata["mirna_sites"]] == [2.0, 0.5]


def test_pyrosetta_interface_scores_attached_structure(monkeypatch):
    config = PyRosettaInterfaceConfig(metric="interface_dG", desired_value=-10.0, tolerance=10.0)
    captured = {}

    monkeypatch.setitem(
        pyrosetta_interface_constraint.__globals__,
        "InterfaceStructureInput",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(
        pyrosetta_interface_constraint.__globals__,
        "PyRosettaInterfaceAnalyzerInput",
        lambda inputs: SimpleNamespace(inputs=inputs),
    )

    def fake_run_pyrosetta_interface_analyzer(tool_input, tool_config):
        captured["tool_input"] = tool_input
        metrics = SimpleNamespace(interface_dG=-5.0, model_dump=lambda mode="json": {"interface_dG": -5.0})
        return SimpleNamespace(results=[metrics])

    monkeypatch.setitem(
        pyrosetta_interface_constraint.__globals__,
        "run_pyrosetta_interface_analyzer",
        fake_run_pyrosetta_interface_analyzer,
    )

    sequence = Sequence("ACDE", "protein")
    sequence.structure = object()
    (output,) = pyrosetta_interface_constraint([(sequence,)], config)

    assert captured["tool_input"].inputs[0]["structure"] is sequence.structure
    assert output.score == 0.5
    assert output.metadata["pyrosetta_interface_value"] == -5.0


def test_alphagenome_splice_junction_scores_requested_pair(monkeypatch):
    config = AlphaGenomeSpliceJunctionConfig(
        genomic_context="A" * 20,
        cassette_left_context="",
        cassette_right_context="",
        ontology_terms=["EFO:0000001"],
        donor_pos=2,
        acceptor_pos=4,
    )
    values = np.zeros((20, 20), dtype=float)
    values[9, 11] = 0.8

    monkeypatch.setitem(
        alphagenome_splice_junction_constraint.__globals__,
        "AlphaGenomePredictSequencesInput",
        lambda sequences: SimpleNamespace(sequences=sequences),
    )
    monkeypatch.setitem(
        alphagenome_splice_junction_constraint.__globals__,
        "AlphaGenomePredictSequencesConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def fake_run_alphagenome_predict_sequences(tool_input, tool_config):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    result={
                        "predictions": {
                            "SPLICE_JUNCTIONS": {
                                "values": values.tolist(),
                            }
                        }
                    }
                )
            ]
        )

    monkeypatch.setitem(
        alphagenome_splice_junction_constraint.__globals__,
        "run_alphagenome_predict_sequences",
        fake_run_alphagenome_predict_sequences,
    )

    left = Sequence("AA", "dna")
    intron = Sequence("GT", "dna")
    right = Sequence("AA", "dna")
    (output,) = alphagenome_splice_junction_constraint([(left, intron, right)], config)

    assert output.score == pytest.approx(0.2)
    assert output.metadata["donor_abs"] == 9
    assert output.metadata["acceptor_abs"] == 11
    assert output.metadata["alphagenome_splice_junction_raw"] == 0.8


def test_alphagenome_splice_junction_scores_metadata_row_for_compact_output(monkeypatch):
    config = AlphaGenomeSpliceJunctionConfig(
        genomic_context="A" * 20,
        cassette_left_context="",
        cassette_right_context="",
        ontology_terms=["EFO:0000001"],
        donor_pos=2,
        acceptor_pos=4,
    )
    values = np.array([[0.1], [0.8], [0.2]], dtype=float)
    metadata_records = [
        {"donor_pos": 2, "acceptor_pos": 4, "name": "wrong_pair"},
        {"donor": 9, "acceptor": 11, "name": "requested_pair"},
        {"donor_pos": 1, "acceptor_pos": 3, "name": "other_pair"},
    ]

    monkeypatch.setitem(
        alphagenome_splice_junction_constraint.__globals__,
        "AlphaGenomePredictSequencesInput",
        lambda sequences: SimpleNamespace(sequences=sequences),
    )
    monkeypatch.setitem(
        alphagenome_splice_junction_constraint.__globals__,
        "AlphaGenomePredictSequencesConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def fake_run_alphagenome_predict_sequences(tool_input, tool_config):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    result={
                        "predictions": {
                            "SPLICE_JUNCTIONS": {
                                "values": values.tolist(),
                                "metadata": {"records": metadata_records},
                            }
                        }
                    }
                )
            ]
        )

    monkeypatch.setitem(
        alphagenome_splice_junction_constraint.__globals__,
        "run_alphagenome_predict_sequences",
        fake_run_alphagenome_predict_sequences,
    )

    left = Sequence("AA", "dna")
    intron = Sequence("GT", "dna")
    right = Sequence("AA", "dna")
    (output,) = alphagenome_splice_junction_constraint([(left, intron, right)], config)

    assert output.score == pytest.approx(0.2)
    assert output.metadata["alphagenome_splice_junction_raw"] == 0.8
    assert output.metadata["selected_junction_name"] == "requested_pair"
    assert output.metadata["selected_junction_row_index"] == 1


def test_splice_target_window_start_centers_and_clamps():
    # HSV-TK construct geometry: 566 + 301 + 569 = 1436 bp target, donor/acceptor
    # at 564/867 must be windowed into a single 1000 bp window.
    start = splice_target_window_start(1436, [564], [867])
    assert start == 215
    assert 0 <= 564 - start < 1000
    assert 0 <= 867 - start < 1000

    # Identity window when the target already matches the model length.
    assert splice_target_window_start(1000, [500]) == 0

    # Positions spanning more than the window cannot be covered.
    with pytest.raises(ValueError):
        splice_target_window_start(2000, [10], [1500])

    # A target shorter than the required window is rejected.
    with pytest.raises(ValueError):
        splice_target_window_start(800, [400])


def test_splice_transformer_intron_boundary_windows_long_target(monkeypatch):
    from proto_tools import SpliceTransformerType

    h1, intron_len, h2 = 566, 301, 569
    donor = h1 - 2  # 564
    acceptor = h1 + intron_len  # 867
    config = SpliceTransformerIntronBoundaryConfig(
        left_context="A" * 4000,
        right_context="T" * 4000,
        donor_pos=donor,
        acceptor_pos=acceptor,
    )
    captured = {}

    def fake_run_splice_transformer(tool_input, tool_config):
        captured["target_seqs"] = tool_input.target_seqs
        prediction = np.zeros((1, 1000, 18), dtype=float)
        prediction[0, 349, SpliceTransformerType.DONOR.value] = 1.0
        prediction[0, 652, SpliceTransformerType.ACCEPTOR.value] = 1.0
        return SimpleNamespace(prediction=prediction.tolist())

    monkeypatch.setitem(
        splice_transformer_intron_boundary.__globals__,
        "run_splice_transformer",
        fake_run_splice_transformer,
    )

    left = Sequence("C" * h1, "dna")
    core = Sequence("G" * intron_len, "dna")
    right = Sequence("A" * h2, "dna")
    (output,) = splice_transformer_intron_boundary([(left, core, right)], config)

    # Real SpliceTransformerInput validation requires exactly 1000 bp targets.
    assert len(captured["target_seqs"][0]) == 1000
    assert output.metadata["target_window_start"] == 215
    assert output.metadata["windowed_donor_pos"] == [349]
    assert output.metadata["windowed_acceptor_pos"] == [652]
    assert output.metadata["donor_score"] == pytest.approx(0.0)
    assert output.metadata["acceptor_score"] == pytest.approx(0.0)
    assert output.score == pytest.approx(0.0)


def test_splice_transformer_boundary_min_reduction_penalizes_dead_donor(monkeypatch):
    from proto_tools import SpliceTransformerType

    h1, intron_len, h2 = 566, 301, 569
    donor = h1 - 2
    acceptor = h1 + intron_len

    def fake_run_splice_transformer(tool_input, tool_config):
        prediction = np.zeros((1, 1000, 18), dtype=float)
        # Strong acceptor, dead donor -- the failure mode observed in the A549 run.
        prediction[0, 652, SpliceTransformerType.ACCEPTOR.value] = 1.0
        return SimpleNamespace(prediction=prediction.tolist())

    monkeypatch.setitem(
        splice_transformer_intron_boundary.__globals__,
        "run_splice_transformer",
        fake_run_splice_transformer,
    )
    left = Sequence("C" * h1, "dna")
    core = Sequence("G" * intron_len, "dna")
    right = Sequence("A" * h2, "dna")
    triple = [(left, core, right)]

    base = dict(left_context="A" * 4000, right_context="T" * 4000, donor_pos=donor, acceptor_pos=acceptor)
    (mean_out,) = splice_transformer_intron_boundary(triple, SpliceTransformerIntronBoundaryConfig(**base, reduction="mean"))
    (min_out,) = splice_transformer_intron_boundary(triple, SpliceTransformerIntronBoundaryConfig(**base, reduction="min"))

    # mean banks half the reward from the acceptor alone; min refuses to.
    assert mean_out.score == pytest.approx(0.5)
    assert min_out.score == pytest.approx(1.0)
    assert min_out.metadata["reduction"] == "min"


def test_splice_transformer_boundary_peak_search_radius_catches_off_by_one(monkeypatch):
    from proto_tools import SpliceTransformerType

    h1, intron_len, h2 = 566, 301, 569
    donor = h1 - 2  # windows to index 349
    acceptor = h1 + intron_len  # windows to index 652

    def fake_run_splice_transformer(tool_input, tool_config):
        prediction = np.zeros((1, 1000, 18), dtype=float)
        # Donor peak one position upstream of the probed index (SpliceAI convention).
        prediction[0, 348, SpliceTransformerType.DONOR.value] = 0.97
        prediction[0, 652, SpliceTransformerType.ACCEPTOR.value] = 0.97
        return SimpleNamespace(prediction=prediction.tolist())

    monkeypatch.setitem(
        splice_transformer_intron_boundary.__globals__,
        "run_splice_transformer",
        fake_run_splice_transformer,
    )
    triple = [(Sequence("C" * h1, "dna"), Sequence("G" * intron_len, "dna"), Sequence("A" * h2, "dna"))]
    base = dict(left_context="A" * 4000, right_context="T" * 4000, donor_pos=donor, acceptor_pos=acceptor, reduction="min")

    (r0,) = splice_transformer_intron_boundary(triple, SpliceTransformerIntronBoundaryConfig(**base, peak_search_radius=0))
    (r2,) = splice_transformer_intron_boundary(triple, SpliceTransformerIntronBoundaryConfig(**base, peak_search_radius=2))

    # radius 0 misses the off-by-one donor (dead); radius 2 catches it.
    assert r0.metadata["donor_score"] == pytest.approx(1.0)
    assert r2.metadata["donor_score"] == pytest.approx(0.03)
    assert r2.metadata["acceptor_score"] == pytest.approx(0.03)


def test_splice_transformer_specificity_windows_long_target(monkeypatch):
    from proto_tools import SPLICE_TISSUE_CHANNEL_INDEX

    h1, intron_len, h2 = 566, 301, 569
    donor = h1 - 2
    acceptor = h1 + intron_len
    config = SpliceTransformerSpecificityConfig(
        left_context="A" * 4000,
        right_context="T" * 4000,
        splice_pos=[donor, acceptor],
        tissue="AVERAGE",
        direction="max",
    )
    captured = {}

    def fake_run_splice_transformer(tool_input, tool_config):
        captured["target_seqs"] = tool_input.target_seqs
        prediction = np.zeros((1, 1000, 18), dtype=float)
        channel_index = SPLICE_TISSUE_CHANNEL_INDEX["AVERAGE"]
        if channel_index is None:
            prediction[0, [349, 652], 3:] = 1.0
        else:
            prediction[0, [349, 652], channel_index] = 1.0
        return SimpleNamespace(prediction=prediction.tolist())

    monkeypatch.setitem(
        splice_transformer_specificity.__globals__,
        "run_splice_transformer",
        fake_run_splice_transformer,
    )

    left = Sequence("C" * h1, "dna")
    core = Sequence("G" * intron_len, "dna")
    right = Sequence("A" * h2, "dna")
    (output,) = splice_transformer_specificity([(left, core, right)], config)

    assert len(captured["target_seqs"][0]) == 1000
    assert output.metadata["target_window_start"] == 215
    assert output.metadata["windowed_splice_pos"] == [349, 652]
    # direction="max" => score = 1 - raw; raw is 1.0 at both scored positions.
    assert output.score == pytest.approx(0.0)
