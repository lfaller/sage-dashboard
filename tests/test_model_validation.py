"""Tests for model validation and benchmarking (Phase 6A.2).

Tests ModelValidator class for validating trained sex classifiers
against benchmark datasets and comparing performance metrics.
"""

import pytest
import tempfile
from pathlib import Path

import numpy as np

from src.sage.model_training import SexClassifierTrainer
from src.sage.model_validation import ModelValidator, ValidationReport, BenchmarkResult


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_initialization_with_required_fields(self):
        """Should initialize with required fields."""
        report = ValidationReport(
            model_name="sex_classifier_v1",
            validation_date="2026-01-25",
            accuracy=0.92,
            precision=0.91,
            recall=0.93,
            f1_score=0.92,
            auc_score=0.95,
            total_samples=100,
            true_positives=46,
            true_negatives=46,
            false_positives=4,
            false_negatives=4,
        )

        assert report.model_name == "sex_classifier_v1"
        assert report.accuracy == 0.92
        assert report.total_samples == 100

    def test_to_dict_serialization(self):
        """Should serialize to dictionary."""
        report = ValidationReport(
            model_name="sex_classifier_v1",
            validation_date="2026-01-25",
            accuracy=0.92,
            precision=0.91,
            recall=0.93,
            f1_score=0.92,
            auc_score=0.95,
            total_samples=100,
            true_positives=46,
            true_negatives=46,
            false_positives=4,
            false_negatives=4,
        )

        data = report.to_dict()

        assert isinstance(data, dict)
        assert data["model_name"] == "sex_classifier_v1"
        assert data["accuracy"] == 0.92

    def test_from_dict_deserialization(self):
        """Should deserialize from dictionary."""
        data = {
            "model_name": "sex_classifier_v1",
            "validation_date": "2026-01-25",
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.93,
            "f1_score": 0.92,
            "auc_score": 0.95,
            "total_samples": 100,
            "true_positives": 46,
            "true_negatives": 46,
            "false_positives": 4,
            "false_negatives": 4,
            "notes": "Test validation",
        }

        report = ValidationReport.from_dict(data)

        assert report.model_name == "sex_classifier_v1"
        assert report.accuracy == 0.92
        assert report.notes == "Test validation"

    def test_specificity_calculation(self):
        """Should calculate specificity from confusion matrix."""
        report = ValidationReport(
            model_name="test",
            validation_date="2026-01-25",
            accuracy=0.90,
            precision=0.90,
            recall=0.90,
            f1_score=0.90,
            auc_score=0.95,
            total_samples=100,
            true_positives=45,
            true_negatives=45,
            false_positives=5,
            false_negatives=5,
        )

        # Specificity = TN / (TN + FP)
        assert report.specificity == pytest.approx(0.9)

    def test_sensitivity_equals_recall(self):
        """Should have sensitivity equal to recall."""
        report = ValidationReport(
            model_name="test",
            validation_date="2026-01-25",
            accuracy=0.90,
            precision=0.90,
            recall=0.92,
            f1_score=0.91,
            auc_score=0.95,
            total_samples=100,
            true_positives=46,
            true_negatives=45,
            false_positives=5,
            false_negatives=4,
        )

        assert report.sensitivity == report.recall


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_initialization(self):
        """Should initialize with benchmark name and metrics."""
        benchmark = BenchmarkResult(
            name="Flynn et al. (2021) Microarray",
            accuracy=0.917,
            source="BMC Bioinformatics",
            platform="Microarray",
            sample_size=10000,
        )

        assert benchmark.name == "Flynn et al. (2021) Microarray"
        assert benchmark.accuracy == 0.917


class TestModelValidator:
    """Test ModelValidator for validation and benchmarking."""

    def test_initialization(self):
        """Should initialize validator."""
        validator = ModelValidator()
        assert validator is not None

    def test_validate_on_test_set(self):
        """Should validate model on test set."""
        # Train a simple model
        trainer = SexClassifierTrainer()
        X_train = np.random.randn(40, 5)
        y_train = np.array([0] * 20 + [1] * 20)
        trainer.train(X_train, y_train)

        # Prepare test set
        X_test = np.random.randn(20, 5)
        y_test = np.array([0] * 10 + [1] * 10)

        # Validate
        validator = ModelValidator()
        report = validator.validate(trainer, X_test, y_test, model_name="test_model")

        assert report.model_name == "test_model"
        assert 0.0 <= report.accuracy <= 1.0
        assert report.total_samples == 20

    def test_validation_metrics_are_valid(self):
        """Should produce valid metrics from validation."""
        trainer = SexClassifierTrainer()
        X_train = np.random.randn(60, 5)
        y_train = np.array([0] * 30 + [1] * 30)
        trainer.train(X_train, y_train)

        X_test = np.random.randn(20, 5)
        y_test = np.array([0] * 10 + [1] * 10)

        validator = ModelValidator()
        report = validator.validate(trainer, X_test, y_test)

        # Check metric ranges
        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.f1_score <= 1.0
        assert 0.0 <= report.auc_score <= 1.0

    def test_confusion_matrix_sums_correctly(self):
        """Should have confusion matrix elements sum to total samples."""
        trainer = SexClassifierTrainer()
        X_train = np.random.randn(60, 5)
        y_train = np.array([0] * 30 + [1] * 30)
        trainer.train(X_train, y_train)

        X_test = np.random.randn(20, 5)
        y_test = np.array([0] * 10 + [1] * 10)

        validator = ModelValidator()
        report = validator.validate(trainer, X_test, y_test)

        total = (
            report.true_positives
            + report.true_negatives
            + report.false_positives
            + report.false_negatives
        )
        assert total == report.total_samples

    def test_compare_against_benchmark(self):
        """Should compare model performance against benchmark."""
        trainer = SexClassifierTrainer()
        X_train = np.random.randn(60, 5)
        y_train = np.array([0] * 30 + [1] * 30)
        trainer.train(X_train, y_train)

        X_test = np.random.randn(20, 5)
        y_test = np.array([0] * 10 + [1] * 10)

        validator = ModelValidator()
        report = validator.validate(trainer, X_test, y_test)

        # Compare against Flynn benchmark
        comparison = validator.compare_benchmark(report, benchmark_name="microarray")

        assert "model_accuracy" in comparison
        assert "benchmark_accuracy" in comparison
        assert "difference" in comparison

    def test_get_flynn_et_al_benchmarks(self):
        """Should have access to Flynn et al. benchmarks."""
        validator = ModelValidator()
        benchmarks = validator.get_benchmarks()

        assert len(benchmarks) > 0
        assert any(b.name for b in benchmarks if "Flynn" in b.name)

    def test_validate_balanced_dataset(self):
        """Should handle balanced validation datasets."""
        # Use seed for reproducibility with balanced data
        np.random.seed(42)

        trainer = SexClassifierTrainer()
        X_train = np.random.randn(100, 5)
        y_train = np.array([0] * 50 + [1] * 50)
        trainer.train(X_train, y_train)

        X_test = np.random.randn(50, 5)
        y_test = np.array([0] * 25 + [1] * 25)

        validator = ModelValidator()
        report = validator.validate(trainer, X_test, y_test)

        # Validation should complete successfully with balanced data
        assert report.total_samples == 50
        assert 0.0 <= report.accuracy <= 1.0

    def test_validate_imbalanced_dataset(self):
        """Should handle imbalanced validation datasets."""
        trainer = SexClassifierTrainer()
        X_train = np.random.randn(100, 5)
        y_train = np.array([0] * 80 + [1] * 20)
        trainer.train(X_train, y_train)

        X_test = np.random.randn(50, 5)
        y_test = np.array([0] * 40 + [1] * 10)

        validator = ModelValidator()
        report = validator.validate(trainer, X_test, y_test)

        assert report.total_samples == 50

    def test_validation_report_json_export(self):
        """Should export validation report as JSON."""
        trainer = SexClassifierTrainer()
        X_train = np.random.randn(40, 5)
        y_train = np.array([0] * 20 + [1] * 20)
        trainer.train(X_train, y_train)

        X_test = np.random.randn(20, 5)
        y_test = np.array([0] * 10 + [1] * 10)

        validator = ModelValidator()
        report = validator.validate(trainer, X_test, y_test, model_name="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "validation_report.json"
            validator.export_report(report, str(report_path))

            assert report_path.exists()

            # Verify JSON is valid
            import json

            with open(report_path) as f:
                data = json.load(f)
            assert data["model_name"] == "test"
