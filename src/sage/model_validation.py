"""Model validation and benchmarking for sex classification.

Validates trained sex classifiers against test sets and compares performance
against published benchmarks (e.g., Flynn et al. 2021 BMC Bioinformatics).

Based on Flynn et al. (2021) BMC Bioinformatics methodology.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Results from model validation on test set."""

    model_name: str
    """Name/identifier of the model being validated."""

    validation_date: str
    """Date of validation (YYYY-MM-DD format)."""

    accuracy: float
    """Overall accuracy on test set."""

    precision: float
    """Precision (positive predictive value)."""

    recall: float
    """Recall/sensitivity (true positive rate)."""

    f1_score: float
    """F1 score (harmonic mean of precision and recall)."""

    auc_score: float
    """Area under ROC curve."""

    total_samples: int
    """Total number of test samples."""

    true_positives: int
    """True positives (correctly predicted male)."""

    true_negatives: int
    """True negatives (correctly predicted female)."""

    false_positives: int
    """False positives (female predicted as male)."""

    false_negatives: int
    """False negatives (male predicted as female)."""

    notes: Optional[str] = None
    """Optional validation notes."""

    @property
    def sensitivity(self) -> float:
        """Sensitivity (true positive rate) - alias for recall."""
        return self.recall

    @property
    def specificity(self) -> float:
        """Specificity (true negative rate)."""
        denominator = self.true_negatives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_negatives / denominator

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationReport":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Published benchmark result for comparison."""

    name: str
    """Benchmark name (e.g., 'Flynn et al. (2021) Microarray')."""

    accuracy: float
    """Reported accuracy."""

    source: str = ""
    """Citation/source."""

    platform: str = ""
    """Data platform (Microarray, RNA-seq, etc.)."""

    sample_size: int = 0
    """Number of samples in benchmark study."""

    notes: str = ""
    """Additional notes about benchmark."""


class ModelValidator:
    """Validates trained sex classifiers and compares against benchmarks."""

    # Flynn et al. (2021) BMC Bioinformatics benchmarks
    FLYNN_BENCHMARKS = [
        BenchmarkResult(
            name="Flynn et al. (2021) Microarray",
            accuracy=0.917,
            source="BMC Bioinformatics, 22:84 (2021)",
            platform="Microarray",
            sample_size=10000,
            notes="Elastic net logistic regression on microarray data",
        ),
        BenchmarkResult(
            name="Flynn et al. (2021) RNA-seq",
            accuracy=0.884,
            source="BMC Bioinformatics, 22:84 (2021)",
            platform="RNA-seq",
            sample_size=5000,
            notes="Elastic net logistic regression on RNA-seq data",
        ),
    ]

    def __init__(self):
        """Initialize model validator."""
        logger.debug("ModelValidator initialized")

    def validate(
        self,
        trainer: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "sex_classifier",
    ) -> ValidationReport:
        """Validate model on test set.

        Args:
            trainer: Trained SexClassifierTrainer instance
            X_test: Test features (n_samples, n_features)
            y_test: Test labels (n_samples,) - 0=female, 1=male

        Returns:
            ValidationReport with metrics
        """
        logger.info(f"Validating model: {model_name} on {len(y_test)} samples")

        # Get predictions
        y_pred = trainer.predict(X_test)
        y_pred_proba = trainer.predict_proba(X_test)

        # Extract probabilities for positive class (male)
        y_proba_male = y_pred_proba[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC handling for edge cases
        try:
            auc = roc_auc_score(y_test, y_proba_male)
        except ValueError:
            logger.warning("Could not calculate ROC-AUC (only one class in test set)")
            auc = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        report = ValidationReport(
            model_name=model_name,
            validation_date=datetime.now().strftime("%Y-%m-%d"),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc_score=float(auc),
            total_samples=len(y_test),
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
        )

        logger.info(
            f"Validation complete: accuracy={report.accuracy:.4f}, "
            f"f1={report.f1_score:.4f}, auc={report.auc_score:.4f}"
        )

        return report

    def get_benchmarks(self) -> List[BenchmarkResult]:
        """Get available benchmark results.

        Returns:
            List of BenchmarkResult from published literature
        """
        return self.FLYNN_BENCHMARKS

    def compare_benchmark(
        self,
        report: ValidationReport,
        benchmark_name: str = "microarray",
    ) -> Dict:
        """Compare validation report against published benchmark.

        Args:
            report: ValidationReport from validation()
            benchmark_name: Which benchmark to compare ('microarray' or 'rnaseq')

        Returns:
            Dictionary with comparison results
        """
        # Find matching benchmark
        benchmark = None
        for b in self.FLYNN_BENCHMARKS:
            if benchmark_name.lower() in b.name.lower():
                benchmark = b
                break

        if benchmark is None:
            logger.warning(f"Benchmark not found: {benchmark_name}")
            return {}

        logger.info(
            f"Comparing against {benchmark.name}: "
            f"model={report.accuracy:.4f}, benchmark={benchmark.accuracy:.4f}"
        )

        difference = report.accuracy - benchmark.accuracy
        percent_diff = (difference / benchmark.accuracy) * 100

        return {
            "model_name": report.model_name,
            "model_accuracy": report.accuracy,
            "benchmark_name": benchmark.name,
            "benchmark_accuracy": benchmark.accuracy,
            "difference": difference,
            "percent_difference": percent_diff,
            "meets_benchmark": report.accuracy >= benchmark.accuracy,
        }

    def export_report(self, report: ValidationReport, path: str) -> None:
        """Export validation report as JSON.

        Args:
            report: ValidationReport to export
            path: Path to write JSON file
        """
        from pathlib import Path

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Validation report exported to {path}")
