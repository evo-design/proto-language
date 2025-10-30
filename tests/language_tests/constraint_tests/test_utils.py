"""
Unit tests for proto_language.utils constraint utility functions.

Tests cover:
1. validate_range() - Range validation for constraint parameters
2. calculate_range_deviation() - Deviation calculation for range-based constraints
3. calculate_percentage_range_deviation() - Deviation for percentage-based ranges
4. calculate_normalized_deviation() - Normalized deviation from target values
"""

import pytest
from proto_language.utils import (
    validate_range,
    calculate_range_deviation,
    calculate_percentage_range_deviation,
    calculate_normalized_deviation,
    MIN_ENERGY,
    MAX_ENERGY,
)


class TestValidateRange:
    """Tests for validate_range() utility function."""
    
    def test_value_within_range(self):
        """Test that values within range pass validation."""
        validate_range(50.0, 0.0, 100.0, "test_param")
        validate_range(0.0, 0.0, 100.0, "test_param")  # Lower boundary
        validate_range(100.0, 0.0, 100.0, "test_param")  # Upper boundary
        validate_range(-5.0, -10.0, 10.0, "test_param")  # Negative ranges
    
    def test_value_below_range(self):
        """Test that values below range raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be between"):
            validate_range(-1.0, 0.0, 100.0, "test_param")
        
        with pytest.raises(ValueError, match="gc_content must be between"):
            validate_range(-10.0, 0.0, 100.0, "gc_content")
    
    def test_value_above_range(self):
        """Test that values above range raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be between"):
            validate_range(101.0, 0.0, 100.0, "test_param")
        
        with pytest.raises(ValueError, match="protein_len must be between"):
            validate_range(1000.0, 0.0, 500.0, "protein_len")
    
    def test_edge_cases(self):
        """Test edge cases for validation."""
        # Zero range (min == max)
        validate_range(0.0, 0.0, 0.0, "zero_range")
        
        # Negative ranges
        validate_range(-50.0, -100.0, 0.0, "negative_range")


class TestCalculateRangeDeviation:
    """Tests for calculate_range_deviation() utility function."""
    
    def test_value_within_range(self):
        """Test that values within range have zero deviation (MIN_ENERGY)."""
        assert calculate_range_deviation(50.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_range_deviation(40.0, 40.0, 60.0) == MIN_ENERGY  # Lower boundary
        assert calculate_range_deviation(60.0, 40.0, 60.0) == MIN_ENERGY  # Upper boundary
        assert calculate_range_deviation(55.5, 50.0, 60.0) == MIN_ENERGY
    
    def test_value_below_range(self):
        """Test deviation calculation for values below range."""
        # actual=30, min=40, max=60 -> deviation = (40-30)/40 = 0.25
        assert abs(calculate_range_deviation(30.0, 40.0, 60.0) - 0.25) < 1e-9
        
        # actual=0, min=40, max=60 -> deviation = (40-0)/40 = 1.0
        assert abs(calculate_range_deviation(0.0, 40.0, 60.0) - 1.0) < 1e-9
        
        # actual=10, min=50, max=100 -> deviation = (50-10)/50 = 0.8
        assert abs(calculate_range_deviation(10.0, 50.0, 100.0) - 0.8) < 1e-9
    
    def test_value_above_range(self):
        """Test deviation calculation for values above range."""
        # actual=70, min=40, max=60 -> deviation = (70-60)/60 = 0.166...
        assert abs(calculate_range_deviation(70.0, 40.0, 60.0) - (10.0/60.0)) < 1e-9
        
        # actual=120, min=40, max=60 -> deviation = (120-60)/60 = 1.0
        assert abs(calculate_range_deviation(120.0, 40.0, 60.0) - 1.0) < 1e-9
        
        # actual=200, min=50, max=100 -> deviation = (200-100)/100 = 1.0 (capped)
        deviation = calculate_range_deviation(200.0, 50.0, 100.0)
        assert abs(deviation - 1.0) < 1e-9
    
    def test_capping_at_max_energy(self):
        """Test that deviation is capped at MAX_ENERGY."""
        # Very large deviation below range
        deviation = calculate_range_deviation(0.0, 100.0, 200.0)
        assert deviation <= MAX_ENERGY
        
        # Very large deviation above range
        deviation = calculate_range_deviation(1000.0, 10.0, 20.0)
        assert deviation <= MAX_ENERGY
    
    def test_edge_cases(self):
        """Test edge cases for range deviation."""
        # Value below range with min=10
        assert calculate_range_deviation(5.0, 10.0, 20.0) == 0.5
        
        # Exact match at boundaries
        assert calculate_range_deviation(10.0, 10.0, 20.0) == MIN_ENERGY
        assert calculate_range_deviation(20.0, 10.0, 20.0) == MIN_ENERGY


class TestCalculatePercentageRangeDeviation:
    """Tests for calculate_percentage_range_deviation() utility function."""
    
    def test_value_within_range(self):
        """Test that values within range have zero deviation (MIN_ENERGY)."""
        assert calculate_percentage_range_deviation(50.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_percentage_range_deviation(40.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_percentage_range_deviation(60.0, 40.0, 60.0) == MIN_ENERGY
    
    def test_value_below_range(self):
        """Test deviation for values below range."""
        # actual=30, min=40, max=60 -> deviation = (40-30)/max(40,1) = 0.25
        assert abs(calculate_percentage_range_deviation(30.0, 40.0, 60.0) - 0.25) < 1e-9
        
        # actual=0, min=50, max=70 -> deviation = (50-0)/max(50,1) = 1.0
        assert abs(calculate_percentage_range_deviation(0.0, 50.0, 70.0) - 1.0) < 1e-9
    
    def test_value_above_range(self):
        """Test deviation for values above range."""
        # actual=70, min=40, max=60 -> deviation = (70-60)/max(100-60,1) = 10/40 = 0.25
        assert abs(calculate_percentage_range_deviation(70.0, 40.0, 60.0) - 0.25) < 1e-9
        
        # actual=100, min=40, max=60 -> deviation = (100-60)/max(100-60,1) = 40/40 = 1.0
        assert abs(calculate_percentage_range_deviation(100.0, 40.0, 60.0) - 1.0) < 1e-9
    
    def test_edge_case_min_zero(self):
        """Test edge case when min_val is 0."""
        # actual=0, min=0, max=50 -> deviation = (0-0)/max(0,1) = 0.0
        assert calculate_percentage_range_deviation(0.0, 0.0, 50.0) == MIN_ENERGY
        
        # actual=5, min=0, max=50 -> within range
        assert calculate_percentage_range_deviation(5.0, 0.0, 50.0) == MIN_ENERGY
    
    def test_edge_case_max_hundred(self):
        """Test edge case when max_val is 100."""
        # actual=105, min=40, max=100 -> deviation = (105-100)/max(0,1) = 5/1 = 1.0 (capped)
        deviation = calculate_percentage_range_deviation(105.0, 40.0, 100.0)
        assert deviation <= MAX_ENERGY
        
        # actual=95, min=50, max=100 -> within range
        assert calculate_percentage_range_deviation(95.0, 50.0, 100.0) == MIN_ENERGY
    
    def test_capping_at_max_energy(self):
        """Test that deviation is capped at MAX_ENERGY."""
        # Very large deviation below
        deviation = calculate_percentage_range_deviation(0.0, 90.0, 95.0)
        assert deviation <= MAX_ENERGY
        
        # Very large deviation above
        deviation = calculate_percentage_range_deviation(100.0, 5.0, 10.0)
        assert deviation <= MAX_ENERGY
    
    def test_full_percentage_range(self):
        """Test with full percentage range (0-100)."""
        assert calculate_percentage_range_deviation(50.0, 0.0, 100.0) == MIN_ENERGY
        assert calculate_percentage_range_deviation(0.0, 0.0, 100.0) == MIN_ENERGY
        assert calculate_percentage_range_deviation(100.0, 0.0, 100.0) == MIN_ENERGY


class TestCalculateNormalizedDeviation:
    """Tests for calculate_normalized_deviation() utility function."""
    
    def test_exact_match(self):
        """Test deviation when actual matches target."""
        assert calculate_normalized_deviation(50.0, 50.0) == MIN_ENERGY
        assert calculate_normalized_deviation(0.0, 0.0) == MIN_ENERGY
        assert calculate_normalized_deviation(100.0, 100.0) == MIN_ENERGY
    
    def test_deviation_below_target(self):
        """Test deviation when actual is below target."""
        # actual=40, target=50 -> deviation = |40-50|/max(50,1) = 10/50 = 0.2
        assert abs(calculate_normalized_deviation(40.0, 50.0) - 0.2) < 1e-9
        
        # actual=25, target=100 -> deviation = |25-100|/max(100,1) = 75/100 = 0.75
        assert abs(calculate_normalized_deviation(25.0, 100.0) - 0.75) < 1e-9
    
    def test_deviation_above_target(self):
        """Test deviation when actual is above target."""
        # actual=60, target=50 -> deviation = |60-50|/max(50,1) = 10/50 = 0.2
        assert abs(calculate_normalized_deviation(60.0, 50.0) - 0.2) < 1e-9
        
        # actual=150, target=100 -> deviation = |150-100|/max(100,1) = 50/100 = 0.5
        assert abs(calculate_normalized_deviation(150.0, 100.0) - 0.5) < 1e-9
    
    def test_capping_at_max_energy(self):
        """Test that deviation is capped at MAX_ENERGY."""
        # Large deviation: actual=200, target=50 -> deviation = 150/50 = 3.0, capped at 1.0
        deviation = calculate_normalized_deviation(200.0, 50.0)
        assert deviation == MAX_ENERGY
        
        # Very large deviation
        deviation = calculate_normalized_deviation(0.0, 100.0)
        assert deviation == MAX_ENERGY
    
    def test_zero_target(self):
        """Test behavior when target is 0 (uses 1 as denominator)."""
        # actual=10, target=0 -> deviation = |10-0|/max(0,1) = 10/1 = 1.0 (capped)
        deviation = calculate_normalized_deviation(10.0, 0.0)
        assert deviation == MAX_ENERGY
        
        # actual=0, target=0 -> deviation = |0-0|/max(0,1) = 0/1 = 0.0
        deviation = calculate_normalized_deviation(0.0, 0.0)
        assert deviation == MIN_ENERGY
    
    def test_symmetry(self):
        """Test that deviation is symmetric around target."""
        # |40-50| should equal |60-50|
        dev_below = calculate_normalized_deviation(40.0, 50.0)
        dev_above = calculate_normalized_deviation(60.0, 50.0)
        assert abs(dev_below - dev_above) < 1e-9
        
        # |75-100| should equal |125-100|
        dev_below = calculate_normalized_deviation(75.0, 100.0)
        dev_above = calculate_normalized_deviation(125.0, 100.0)
        assert abs(dev_below - dev_above) < 1e-9
    
    def test_negative_values(self):
        """Test with negative values (uses 1 as denominator due to max(target, 1))."""
        # actual=-10, target=-20 -> deviation = |-10-(-20)|/max(-20,1) = 10/1 = 1.0 (capped)
        deviation = calculate_normalized_deviation(-10.0, -20.0)
        assert deviation <= MAX_ENERGY
        
        # actual=-30, target=-10 -> deviation = |-30-(-10)|/max(-10,1) = 20/1 = 1.0 (capped)
        deviation = calculate_normalized_deviation(-30.0, -10.0)
        assert deviation <= MAX_ENERGY
    
    def test_small_deviations(self):
        """Test that small deviations are calculated accurately."""
        # actual=50.5, target=50.0 -> deviation = 0.5/50 = 0.01
        assert abs(calculate_normalized_deviation(50.5, 50.0) - 0.01) < 1e-9
        
        # actual=99.5, target=100.0 -> deviation = 0.5/100 = 0.005
        assert abs(calculate_normalized_deviation(99.5, 100.0) - 0.005) < 1e-9

