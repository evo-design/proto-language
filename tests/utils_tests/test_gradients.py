"""Tests for proto_language.utils.gradients module."""

import numpy as np

from proto_language.utils.gradients import MGDAMerger, align_norms, normalize_gradient


class TestMGDAMerger:
    def test_equal_gradients_preserved(self) -> None:
        g = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert np.allclose(MGDAMerger().merge([g, g]), g)

    def test_opposing_gradients_cancel(self) -> None:
        g = np.array([[1.0, -1.0], [2.0, 0.5]])
        assert np.linalg.norm(MGDAMerger().merge([g, -g])) < 1e-6


class TestAlignNorms:
    def test_default_eps_does_not_zero(self) -> None:
        large = np.array([3.0, 4.0])
        near_zero = np.array([1e-8, 1e-9])
        aligned = align_norms([large, near_zero], mode="match_first")
        assert not np.allclose(aligned[1], 0.0)

    def test_match_first_zeros_near_zero_gradient(self) -> None:
        large = np.array([3.0, 4.0])
        near_zero = np.array([1e-8, 1e-9])
        aligned = align_norms([large, near_zero], mode="match_first", zero_norm_eps=1e-4)
        assert np.array_equal(aligned[0], large)
        assert np.allclose(aligned[1], 0.0)

    def test_match_first_preserves_direction(self) -> None:
        large = np.array([3.0, 4.0])
        normal = np.array([1.0, 0.0])
        aligned = align_norms([large, normal], mode="match_first", zero_norm_eps=1e-4)
        assert np.isclose(np.linalg.norm(aligned[1]), np.linalg.norm(large), rtol=1e-5)
        assert np.allclose(aligned[1] / np.linalg.norm(aligned[1]), normal / np.linalg.norm(normal), atol=1e-6)

    def test_match_first_zero_reference_returns_unmodified(self) -> None:
        zero = np.array([0.0, 0.0])
        normal = np.array([1.0, 2.0])
        aligned = align_norms([zero, normal], mode="match_first", zero_norm_eps=1e-4)
        assert np.allclose(aligned[0], zero)
        assert np.allclose(aligned[1], normal)


class TestNormalizeGradient:
    def test_sqrt_length_formula_matches_germinal(self) -> None:
        """Verify ``g * sqrt(eff_L) / ||g||`` where ``eff_L`` counts positions with non-zero norm."""
        grad = np.zeros((5, 20))
        grad[0, 3] = 1.0
        grad[2, 7] = 2.0  # eff_L=2
        expected = grad * np.sqrt(2.0) / (np.linalg.norm(grad) + 1e-7)
        assert np.allclose(normalize_gradient(grad, mode="sqrt_length"), expected)

    def test_unit_vs_sqrt_length_differ(self) -> None:
        grad = np.random.default_rng(0).standard_normal((8, 20))
        unit = normalize_gradient(grad, mode="unit")
        sqrtl = normalize_gradient(grad, mode="sqrt_length")
        assert np.isclose(np.linalg.norm(unit), 1.0)
        assert np.linalg.norm(sqrtl) > 1.0
        assert not np.allclose(unit, sqrtl)
