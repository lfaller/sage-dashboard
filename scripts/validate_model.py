#!/usr/bin/env python3
"""Validate trained sex classifier model.

Evaluates a trained model on test data and compares against benchmarks.

Usage:
    poetry run python scripts/validate_model.py \\
        --model-path models/sex_classifier/1.0.0 \\
        --test-data data/test.csv \\
        --output validation_report.json \\
        --benchmark microarray

Based on Flynn et al. (2021) BMC Bioinformatics methodology.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.sage.logging_config import get_logger
from src.sage.model_persistence import ModelPersistence
from src.sage.model_validation import ModelValidator

logger = get_logger(__name__)


def load_test_data(data_path: str) -> tuple:
    """Load test data from CSV.

    Args:
        data_path: Path to CSV file with columns: sample_id, sex, and feature columns

    Returns:
        Tuple of (X, y, sample_ids) where:
        - X: Feature matrix (n_samples, n_features)
        - y: Labels (n_samples,) - 0=female, 1=male
        - sample_ids: List of sample identifiers
    """
    logger.info(f"Loading test data from {data_path}")

    X = []
    y = []
    sample_ids = []

    with open(data_path) as f:
        reader = csv.DictReader(f)

        feature_cols = None

        for row in reader:
            if feature_cols is None:
                feature_cols = [col for col in reader.fieldnames if col not in ("sample_id", "sex")]

            sample_ids.append(row["sample_id"])

            sex = row["sex"].lower()
            if sex in ("male", "m", "1"):
                y.append(1)
            elif sex in ("female", "f", "0"):
                y.append(0)
            else:
                logger.warning(f"Unknown sex value: {sex}")
                continue

            try:
                features = [float(row[col]) for col in feature_cols]
                X.append(features)
            except (ValueError, KeyError) as e:
                logger.error(f"Error parsing features for {row['sample_id']}: {e}")
                continue

    if not X:
        raise ValueError("No valid test data found")

    logger.info(f"Loaded {len(X)} test samples with {len(feature_cols)} features")

    return np.array(X), np.array(y), sample_ids


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate trained sex classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model directory (e.g., models/sex_classifier/1.0.0)",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--output",
        help="Path to save validation report JSON",
    )
    parser.add_argument(
        "--benchmark",
        choices=["microarray", "rnaseq"],
        default="microarray",
        help="Benchmark to compare against (default: microarray)",
    )
    parser.add_argument(
        "--model-name",
        default="sex_classifier",
        help="Model name for report (default: sex_classifier)",
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 60)
        logger.info("Validating Sex Classifier Model")
        logger.info("=" * 60)

        # Parse model path
        model_parts = args.model_path.rstrip("/").split("/")
        if len(model_parts) < 2:
            raise ValueError(f"Invalid model path: {args.model_path}")

        model_version = model_parts[-1]
        model_name = model_parts[-2]
        model_base = "/".join(model_parts[:-2])

        logger.info(f"\nLoading model: {model_name}/{model_version}")

        # Load model
        persistence = ModelPersistence()
        trainer = persistence.load_model(
            Path(model_base),
            name=model_name,
            version=model_version,
        )

        # Load test data
        X_test, y_test, sample_ids = load_test_data(args.test_data)

        # Validate
        logger.info(f"\nValidating on {len(y_test)} test samples...")
        validator = ModelValidator()
        report = validator.validate(
            trainer,
            X_test,
            y_test,
            model_name=args.model_name,
        )

        logger.info("\nValidation Results:")
        logger.info(f"  Accuracy: {report.accuracy:.4f}")
        logger.info(f"  Precision: {report.precision:.4f}")
        logger.info(f"  Recall: {report.recall:.4f}")
        logger.info(f"  Specificity: {report.specificity:.4f}")
        logger.info(f"  F1 Score: {report.f1_score:.4f}")
        logger.info(f"  AUC Score: {report.auc_score:.4f}")
        logger.info("\nConfusion Matrix:")
        logger.info(f"  True Positives: {report.true_positives}")
        logger.info(f"  True Negatives: {report.true_negatives}")
        logger.info(f"  False Positives: {report.false_positives}")
        logger.info(f"  False Negatives: {report.false_negatives}")

        # Compare benchmark
        logger.info(f"\nComparing against Flynn et al. benchmark ({args.benchmark})...")
        comparison = validator.compare_benchmark(report, benchmark_name=args.benchmark)

        if comparison:
            logger.info(f"  Benchmark Accuracy: {comparison['benchmark_accuracy']:.4f}")
            logger.info(f"  Model Accuracy: {comparison['model_accuracy']:.4f}")
            logger.info(f"  Difference: {comparison['difference']:+.4f}")
            logger.info(f"  Percent Difference: {comparison['percent_difference']:+.2f}%")

            if comparison["meets_benchmark"]:
                logger.info("  ✓ Model MEETS benchmark")
            else:
                logger.info("  ⚠ Model below benchmark")

        # Save report if requested
        if args.output:
            logger.info(f"\nSaving report to {args.output}")
            validator.export_report(report, args.output)

            # Also save comparison
            output_path = Path(args.output)
            comparison_path = output_path.parent / (output_path.stem + "_comparison.json")
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("✓ Validation complete")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
