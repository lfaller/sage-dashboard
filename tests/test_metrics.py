"""Tests for metrics calculations."""
import pytest
from src.sage.metrics import calculate_completeness_percentage


def test_calculate_completeness_percentage_basic():
    """Test basic percentage calculation."""
    assert calculate_completeness_percentage(50, 100) == 50.0


def test_calculate_completeness_percentage_full():
    """Test 100% completeness."""
    assert calculate_completeness_percentage(100, 100) == 100.0


def test_calculate_completeness_percentage_zero():
    """Test 0% completeness."""
    assert calculate_completeness_percentage(0, 100) == 0.0


def test_calculate_completeness_percentage_decimal():
    """Test decimal precision."""
    result = calculate_completeness_percentage(87123, 127432)
    assert abs(result - 68.4) < 0.1  # Approximately 68.4%


def test_calculate_completeness_percentage_zero_total():
    """Test edge case with zero total."""
    assert calculate_completeness_percentage(0, 0) == 0.0


def test_calculate_completeness_percentage_negative_values():
    """Test that negative values raise error."""
    with pytest.raises(ValueError):
        calculate_completeness_percentage(-10, 100)

    with pytest.raises(ValueError):
        calculate_completeness_percentage(10, -100)
