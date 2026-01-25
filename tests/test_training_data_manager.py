"""Tests for training data extraction and management (Phase 6A.2).

Tests TrainingDataExtractor and TrainingDataset classes for extracting
high-confidence sex labels from SAGE studies for model training.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.sage.training_data_manager import (
    TrainingDataExtractor,
    TrainingDataset,
    TrainingDatasetMetadata,
)


class TestTrainingDataExtractor:
    """Test TrainingDataExtractor for extracting training labels."""

    def test_initialization(self):
        """Should initialize with database client."""
        extractor = TrainingDataExtractor()
        assert extractor is not None

    def test_fetch_high_confidence_samples_requires_threshold(self):
        """Should fetch samples with confidence >= threshold."""
        extractor = TrainingDataExtractor()

        # Mock database response
        with patch.object(extractor, "fetch_from_database") as mock_fetch:
            mock_fetch.return_value = [
                {"gsm_id": "GSM001", "sex": "male", "confidence": 0.95},
                {"gsm_id": "GSM002", "sex": "female", "confidence": 0.92},
                {"gsm_id": "GSM003", "sex": "male", "confidence": 0.85},
            ]

            samples = extractor.fetch_high_confidence_samples(threshold=0.90)

            assert len(samples) == 2
            assert samples[0]["gsm_id"] == "GSM001"
            assert samples[1]["gsm_id"] == "GSM002"

    def test_fetch_samples_preserves_metadata(self):
        """Should preserve source and confidence information."""
        extractor = TrainingDataExtractor()

        with patch.object(extractor, "fetch_from_database") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "gsm_id": "GSM001",
                    "sex": "male",
                    "confidence": 0.95,
                    "source": "characteristics",
                    "gse_id": "GSE123",
                }
            ]

            samples = extractor.fetch_high_confidence_samples(threshold=0.90)

            assert samples[0]["source"] == "characteristics"
            assert samples[0]["gse_id"] == "GSE123"
            assert samples[0]["confidence"] == 0.95

    def test_validate_label_consistency_detects_conflicts(self):
        """Should detect conflicting sex labels for same sample."""
        extractor = TrainingDataExtractor()

        conflicting_samples = [
            {"gsm_id": "GSM001", "sex": "male", "source": "characteristics"},
            {"gsm_id": "GSM001", "sex": "female", "source": "sample_names"},
        ]

        conflicts = extractor.validate_label_consistency(conflicting_samples)

        assert len(conflicts) > 0
        assert conflicts[0]["gsm_id"] == "GSM001"

    def test_validate_label_consistency_allows_same_labels(self):
        """Should allow duplicate labels if they agree."""
        extractor = TrainingDataExtractor()

        consistent_samples = [
            {"gsm_id": "GSM001", "sex": "male", "source": "characteristics"},
            {"gsm_id": "GSM001", "sex": "male", "source": "sample_names"},
        ]

        conflicts = extractor.validate_label_consistency(consistent_samples)

        assert len(conflicts) == 0

    def test_resolve_conflicting_labels_prefers_characteristics(self):
        """Should prefer characteristics over sample names."""
        extractor = TrainingDataExtractor()

        samples = [
            {
                "gsm_id": "GSM001",
                "sex": "male",
                "source": "characteristics",
                "confidence": 0.95,
            },
            {
                "gsm_id": "GSM001",
                "sex": "female",
                "source": "sample_names",
                "confidence": 0.60,
            },
        ]

        resolved = extractor.resolve_conflicting_labels(samples)

        assert len(resolved) == 1
        assert resolved[0]["sex"] == "male"
        assert resolved[0]["source"] == "characteristics"

    def test_export_training_fixture_creates_valid_json(self):
        """Should export training data as valid JSON fixture."""
        extractor = TrainingDataExtractor()

        training_data = [
            {
                "gsm_id": "GSM001",
                "gse_id": "GSE123",
                "sex": "male",
                "confidence": 0.95,
                "source": "characteristics",
            },
            {
                "gsm_id": "GSM002",
                "gse_id": "GSE123",
                "sex": "female",
                "confidence": 0.92,
                "source": "characteristics",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            extractor.export_training_fixture(training_data, output_path, version="1.0.0")

            with open(output_path, "r") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "samples" in data
            assert data["metadata"]["version"] == "1.0.0"
            assert len(data["samples"]) == 2
        finally:
            Path(output_path).unlink()

    def test_compute_dataset_statistics(self):
        """Should compute statistics on training dataset."""
        extractor = TrainingDataExtractor()

        training_data = [
            {"gsm_id": f"GSM{i:03d}", "sex": "male" if i % 2 == 0 else "female"} for i in range(20)
        ]

        stats = extractor.compute_dataset_statistics(training_data)

        assert stats["total_samples"] == 20
        assert stats["male_count"] == 10
        assert stats["female_count"] == 10
        assert stats["balance_ratio"] == pytest.approx(0.5)

    def test_compute_statistics_handles_imbalance(self):
        """Should correctly handle imbalanced datasets."""
        extractor = TrainingDataExtractor()

        training_data = [
            {"gsm_id": "GSM001", "sex": "male"},
            {"gsm_id": "GSM002", "sex": "male"},
            {"gsm_id": "GSM003", "sex": "male"},
            {"gsm_id": "GSM004", "sex": "female"},
        ]

        stats = extractor.compute_dataset_statistics(training_data)

        assert stats["male_count"] == 3
        assert stats["female_count"] == 1
        assert stats["balance_ratio"] == pytest.approx(0.75)

    def test_handle_too_few_samples_warning(self):
        """Should warn if training set too small (<20 samples)."""
        extractor = TrainingDataExtractor()

        small_data = [
            {"gsm_id": f"GSM{i:03d}", "sex": "male" if i % 2 == 0 else "female"} for i in range(10)
        ]

        with pytest.warns(UserWarning, match="fewer than 20 samples"):
            extractor.validate_training_set_size(small_data)


class TestTrainingDataset:
    """Test TrainingDataset dataclass."""

    def test_initialization(self):
        """Should initialize with samples and labels."""
        samples = ["GSM001", "GSM002", "GSM003", "GSM004"]
        labels = [1, 0, 1, 0]  # 1=male, 0=female

        dataset = TrainingDataset(samples=samples, labels=np.array(labels))

        assert len(dataset.samples) == 4
        assert len(dataset.labels) == 4

    def test_properties_male_female_counts(self):
        """Should calculate male and female counts."""
        samples = ["GSM001", "GSM002", "GSM003"]
        labels = np.array([1, 0, 1])  # 2 male, 1 female

        dataset = TrainingDataset(samples=samples, labels=labels)

        assert dataset.male_count == 2
        assert dataset.female_count == 1
        assert dataset.total_samples == 3

    def test_balance_ratio(self):
        """Should compute balance ratio."""
        samples = ["GSM001", "GSM002", "GSM003", "GSM004"]
        labels = np.array([1, 1, 0, 0])  # Perfect balance

        dataset = TrainingDataset(samples=samples, labels=labels)

        assert dataset.balance_ratio == pytest.approx(0.5)

    def test_to_numpy_returns_labels_array(self):
        """Should return labels as numpy array."""
        samples = ["GSM001", "GSM002"]
        labels = np.array([1, 0])

        dataset = TrainingDataset(samples=samples, labels=labels)

        y = dataset.to_labels_array()

        assert isinstance(y, np.ndarray)
        assert np.array_equal(y, labels)

    def test_stratified_split_maintains_balance(self):
        """Should maintain class balance in train/test split."""
        samples = [f"GSM{i:03d}" for i in range(20)]
        labels = np.array([1, 0] * 10)  # Balanced: 10M, 10F

        dataset = TrainingDataset(samples=samples, labels=labels)
        train_dataset, test_dataset = dataset.stratified_split(test_size=0.2)

        assert train_dataset.total_samples == 16
        assert test_dataset.total_samples == 4

        # Check balance maintained
        train_ratio = train_dataset.male_count / train_dataset.total_samples
        test_ratio = test_dataset.male_count / test_dataset.total_samples

        assert train_ratio == pytest.approx(0.5, abs=0.1)
        assert test_ratio == pytest.approx(0.5, abs=0.1)

    def test_stratified_split_on_imbalanced_data(self):
        """Should maintain imbalance ratio in both splits."""
        samples = [f"GSM{i:03d}" for i in range(30)]
        labels = np.array([1] * 20 + [0] * 10)  # Imbalanced: 20M, 10F (67/33)

        dataset = TrainingDataset(samples=samples, labels=labels)
        train_dataset, test_dataset = dataset.stratified_split(test_size=0.2)

        # Both splits should maintain ~67/33 ratio
        train_ratio = train_dataset.male_count / train_dataset.total_samples
        test_ratio = test_dataset.male_count / test_dataset.total_samples

        assert train_ratio == pytest.approx(0.667, abs=0.1)
        assert test_ratio == pytest.approx(0.667, abs=0.1)


class TestTrainingDatasetMetadata:
    """Test TrainingDatasetMetadata for fixture metadata."""

    def test_initialization_with_required_fields(self):
        """Should initialize with required fields."""
        metadata = TrainingDatasetMetadata(
            name="sage_training_v1",
            version="1.0.0",
            total_samples=40,
            male_count=20,
            female_count=20,
        )

        assert metadata.name == "sage_training_v1"
        assert metadata.version == "1.0.0"
        assert metadata.total_samples == 40

    def test_to_dict_serialization(self):
        """Should serialize to dictionary."""
        metadata = TrainingDatasetMetadata(
            name="sage_training_v1",
            version="1.0.0",
            total_samples=40,
            male_count=20,
            female_count=20,
            description="Test training set",
        )

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "sage_training_v1"
        assert data["description"] == "Test training set"

    def test_from_dict_deserialization(self):
        """Should deserialize from dictionary."""
        data = {
            "name": "sage_training_v1",
            "version": "1.0.0",
            "total_samples": 40,
            "male_count": 20,
            "female_count": 20,
        }

        metadata = TrainingDatasetMetadata.from_dict(data)

        assert metadata.name == "sage_training_v1"
        assert metadata.version == "1.0.0"
