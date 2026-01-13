"""Metrics and calculations for SAGE Dashboard."""


def calculate_completeness_percentage(count: int, total: int) -> float:
    """
    Calculate metadata completeness percentage.

    Args:
        count: Number of items with metadata
        total: Total number of items

    Returns:
        Percentage as float (0-100)

    Raises:
        ValueError: If count or total is negative
    """
    if count < 0 or total < 0:
        raise ValueError("Count and total must be non-negative")

    if total == 0:
        return 0.0

    return (count / total) * 100
