"""Tests for training fixtures and fixture loading utilities.

Tests fixture loading, metadata retrieval, and proper data distribution
for reproducible model training.
"""


import numpy as np
import pytest

from tests.fixtures.training_fixtures import (
    load_training_fixture,
    get_fixture_metadata,
    list_available_fixtures,
)


class TestTrainingFixtures:
    """Test training fixture loading and validation."""

    def test_load_tiny_fixture_returns_correct_shape(self):
        """Should load tiny fixture with correct shape."""
        X, y = load_training_fixture("tiny")

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 10  # 10 samples
        assert X.shape[1] == 20  # 20 features
        assert y.shape == (10,)

    def test_load_balanced_fixture_returns_correct_shape(self):
        """Should load balanced fixture with correct shape."""
        X, y = load_training_fixture("v1")

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 100  # 100 samples
        assert X.shape[1] == 100  # 100 features
        assert y.shape == (100,)

    def test_load_imbalanced_fixture_returns_correct_shape(self):
        """Should load imbalanced fixture with correct shape."""
        X, y = load_training_fixture("imbalanced")

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 60  # 60 samples
        assert X.shape[1] == 100  # 100 features
        assert y.shape == (60,)

    def test_tiny_fixture_has_balanced_distribution(self):
        """Should have balanced 5F/5M distribution in tiny fixture."""
        X, y = load_training_fixture("tiny")

        n_female = np.sum(y == 0)
        n_male = np.sum(y == 1)

        assert n_female == 5
        assert n_male == 5

    def test_v1_fixture_has_balanced_distribution(self):
        """Should have balanced 50F/50M distribution in v1 fixture."""
        X, y = load_training_fixture("v1")

        n_female = np.sum(y == 0)
        n_male = np.sum(y == 1)

        assert n_female == 50
        assert n_male == 50

    def test_imbalanced_fixture_has_correct_distribution(self):
        """Should have 75M/25F distribution in imbalanced fixture."""
        X, y = load_training_fixture("imbalanced")

        n_female = np.sum(y == 0)
        n_male = np.sum(y == 1)

        assert n_female == 15
        assert n_male == 45

    def test_fixture_data_is_numeric(self):
        """Should return numeric data with valid values."""
        X, y = load_training_fixture("tiny")

        assert X.dtype in [np.float32, np.float64]
        assert y.dtype in [np.int32, np.int64, int]
        assert np.isfinite(X).all()  # No NaN or inf
        assert np.all((y == 0) | (y == 1))  # Only 0 or 1

    def test_fixture_data_has_reasonable_distribution(self):
        """Should have expression values in reasonable range."""
        X, y = load_training_fixture("v1")

        # Expression values should be in log2 scale (typically -5 to 15)
        assert X.min() >= -5
        assert X.max() <= 20

    def test_get_tiny_fixture_metadata(self):
        """Should retrieve metadata for tiny fixture."""
        metadata = get_fixture_metadata("tiny")

        assert metadata["name"] == "tiny"
        assert metadata["n_samples"] == 10
        assert metadata["n_features"] == 20
        assert metadata["n_female"] == 5
        assert metadata["n_male"] == 5
        assert metadata["balanced"] is True

    def test_get_v1_fixture_metadata(self):
        """Should retrieve metadata for v1 fixture."""
        metadata = get_fixture_metadata("v1")

        assert metadata["name"] == "v1"
        assert metadata["n_samples"] == 100
        assert metadata["n_features"] == 100
        assert metadata["n_female"] == 50
        assert metadata["n_male"] == 50
        assert metadata["balanced"] is True

    def test_get_imbalanced_fixture_metadata(self):
        """Should retrieve metadata for imbalanced fixture."""
        metadata = get_fixture_metadata("imbalanced")

        assert metadata["name"] == "imbalanced"
        assert metadata["n_samples"] == 60
        assert metadata["n_features"] == 100
        assert metadata["n_female"] == 15
        assert metadata["n_male"] == 45
        assert metadata["balanced"] is False

    def test_list_available_fixtures(self):
        """Should list all available fixtures."""
        fixtures = list_available_fixtures()

        assert isinstance(fixtures, list)
        assert len(fixtures) >= 3
        assert "tiny" in fixtures
        assert "v1" in fixtures
        assert "imbalanced" in fixtures

    def test_load_nonexistent_fixture_raises_error(self):
        """Should raise error for non-existent fixture."""
        with pytest.raises(FileNotFoundError):
            load_training_fixture("nonexistent")

    def test_get_metadata_nonexistent_fixture_raises_error(self):
        """Should raise error for non-existent fixture metadata."""
        with pytest.raises(FileNotFoundError):
            get_fixture_metadata("nonexistent")

    def test_fixture_reproducibility(self):
        """Should load same data on repeated calls."""
        X1, y1 = load_training_fixture("tiny")
        X2, y2 = load_training_fixture("tiny")

        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_fixtures_are_independent(self):
        """Fixtures should be independent datasets."""
        X_tiny, y_tiny = load_training_fixture("tiny")
        X_v1, y_v1 = load_training_fixture("v1")
        X_imbalanced, y_imbalanced = load_training_fixture("imbalanced")

        # Different sizes
        assert X_tiny.shape != X_v1.shape
        assert X_v1.shape != X_imbalanced.shape

        # Different data (not just subsets)
        assert not np.array_equal(X_tiny[:10], X_v1[:10])
