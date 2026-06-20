"""tests/language_tests/constraint_tests/test_sequence_annotation/test_targetscan_site_constraint.py."""

import pytest
from pydantic import ValidationError

from proto_language.constraint import targetscan_site_constraint
from proto_language.constraint.sequence_annotation.targetscan_site_constraint import (
    TargetScanSiteConfig,
    _find_sites,
)
from proto_language.core import Constraint, Segment

# miRNA "AACGTACG": pos1=A, seed(2-7)="ACGTAC", m8(pos8)="G".
# core6 = revcomp("ACGTAC") = "GTACGT"; m8-pairing base = complement("G") = "C".
# 8mer site in target = "C" + "GTACGT" + "A" = "CGTACGTA".
MIRNA = "AACGTACG"


def _score(sequence: str, **cfg_kwargs) -> float:
    segment = Segment(sequence=sequence, sequence_type="dna")
    config = TargetScanSiteConfig(mirna_queries=[MIRNA], **cfg_kwargs)
    constraint = Constraint(inputs=[segment], function=targetscan_site_constraint, function_config=config)
    return constraint.evaluate()[0]


class TestTargetScanSiteTyping:
    def test_8mer_detected(self):
        sites = _find_sites("TTTCGTACGTATTT", MIRNA, include_6mer=True)
        assert [s["site_type"] for s in sites] == ["8mer"]

    def test_7mer_m8_detected(self):
        # m8 match (preceding C) but no A1 (next base T).
        sites = _find_sites("TTTCGTACGTTTT", MIRNA, include_6mer=True)
        assert [s["site_type"] for s in sites] == ["7mer-m8"]

    def test_7mer_a1_detected(self):
        # A1 match (next base A) but no m8 (preceding base not C).
        sites = _find_sites("TTAGTACGTATTT", MIRNA, include_6mer=True)
        assert [s["site_type"] for s in sites] == ["7mer-A1"]

    def test_6mer_detected_and_excludable(self):
        # Neither m8 (preceding A) nor A1 (next C).
        seq = "AAGTACGTCC"
        assert [s["site_type"] for s in _find_sites(seq, MIRNA, include_6mer=True)] == ["6mer"]
        assert _find_sites(seq, MIRNA, include_6mer=False) == []


class TestTargetScanScoring:
    def test_8mer_maximize_score(self):
        # one 8mer (weight 1.0); repression 1.0; threshold 2.0 -> bounded 0.5; maximize -> 0.5
        assert _score("TTTCGTACGTATTT", direction="maximize") == pytest.approx(0.5)

    def test_8mer_minimize_score(self):
        assert _score("TTTCGTACGTATTT", direction="minimize") == pytest.approx(0.5)

    def test_no_site_maximize_is_worst(self):
        assert _score("TTTTTTTTTTTT", direction="maximize") == pytest.approx(1.0)

    def test_no_site_minimize_is_best(self):
        assert _score("TTTTTTTTTTTT", direction="minimize") == pytest.approx(0.0)

    def test_7mer_m8_weight_applied(self):
        # one 7mer-m8 (weight 0.8); repression 0.8 / 2.0 = 0.4; maximize -> 0.6
        assert _score("TTTCGTACGTTTT", direction="maximize") == pytest.approx(0.6)

    def test_rna_target_equivalent_to_dna(self):
        segment = Segment(sequence="UUUCGUACGUAUUU", sequence_type="rna")
        config = TargetScanSiteConfig(mirna_queries=[MIRNA], direction="maximize")
        constraint = Constraint(inputs=[segment], function=targetscan_site_constraint, function_config=config)
        assert constraint.evaluate()[0] == pytest.approx(0.5)

    def test_metadata_propagation(self):
        segment = Segment(sequence="TTTCGTACGTATTT", sequence_type="dna")
        config = TargetScanSiteConfig(mirna_queries=[MIRNA], mirna_ids=["test-mir"], direction="maximize")
        Constraint(inputs=[segment], function=targetscan_site_constraint, function_config=config).evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["targetscan_site_constraint"]["data"]
        assert data["targetscan_num_sites"] == 1
        assert data["targetscan_sites"][0]["site_type"] == "8mer"
        assert data["targetscan_sites"][0]["mirna_id"] == "test-mir"

    def test_wrong_sequence_type(self):
        segment = Segment(sequence="MVLSPADKTNVK", sequence_type="protein")
        config = TargetScanSiteConfig(mirna_queries=[MIRNA])
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(inputs=[segment], function=targetscan_site_constraint, function_config=config)


class TestTargetScanConfigValidation:
    def test_short_mirna_raises(self):
        with pytest.raises(ValidationError, match="shorter than 8 nt"):
            TargetScanSiteConfig(mirna_queries=["ACGUAC"])

    def test_mismatched_ids_raises(self):
        with pytest.raises(ValidationError, match="mirna_ids must match"):
            TargetScanSiteConfig(mirna_queries=[MIRNA], mirna_ids=["a", "b"])

    def test_negative_weight_raises(self):
        with pytest.raises(ValidationError, match="cannot contain negative"):
            TargetScanSiteConfig(mirna_queries=[MIRNA], mirna_weights=[-1.0])

    def test_unknown_site_type_weight_raises(self):
        with pytest.raises(ValidationError, match="unknown site types"):
            TargetScanSiteConfig(mirna_queries=[MIRNA], site_type_weights={"9mer": 1.0})

    def test_site_type_weight_override_merges_defaults(self):
        cfg = TargetScanSiteConfig(mirna_queries=[MIRNA], site_type_weights={"8mer": 2.0})
        assert cfg.site_type_weights["8mer"] == 2.0
        assert cfg.site_type_weights["6mer"] == 0.3  # default preserved
