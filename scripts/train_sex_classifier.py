#!/usr/bin/env python3
"""Train elastic net sex classifier model.

Trains a sex classifier on provided expression data with cross-validation
and saves model with versioning.

Usage:
    poetry run python scripts/train_sex_classifier.py \\
        --data data/training.csv \\
        --output models/ \\
        --version 1.0.0 \\
        --cv-folds 10 \\
        --alpha 0.5 \\
        --l1-ratio 0.5

Based on Flynn et al. (2021) BMC Bioinformatics methodology.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from src.sage.logging_config import get_logger
from src.sage.model_training import SexClassifierTrainer, ModelConfig
from src.sage.model_persistence import ModelPersistence

logger = get_logger(__name__)


def load_training_data(data_path: str) -> tuple:
    """Load training data from CSV.

    Args:
        data_path: Path to CSV file with columns: sample_id, sex, and feature columns

    Returns:
        Tuple of (X, y, sample_ids) where:
        - X: Feature matrix (n_samples, n_features)
        - y: Labels (n_samples,) - 0=female, 1=male
        - sample_ids: List of sample identifiers
    """
    logger.info(f"Loading training data from {data_path}")

    X = []
    y = []
    sample_ids = []

    with open(data_path) as f:
        reader = csv.DictReader(f)

        # Get feature column names (all except sample_id and sex)
        feature_cols = None

        for row in reader:
            if feature_cols is None:
                feature_cols = [col for col in reader.fieldnames if col not in ("sample_id", "sex")]

            sample_ids.append(row["sample_id"])

            # Parse sex label
            sex = row["sex"].lower()
            if sex in ("male", "m", "1"):
                y.append(1)
            elif sex in ("female", "f", "0"):
                y.append(0)
            else:
                logger.warning(f"Unknown sex value: {sex}")
                continue

            # Parse features
            try:
                features = [float(row[col]) for col in feature_cols]
                X.append(features)
            except (ValueError, KeyError) as e:
                logger.error(f"Error parsing features for {row['sample_id']}: {e}")
                continue

    if not X:
        raise ValueError("No valid training data found")

    logger.info(f"Loaded {len(X)} samples with {len(feature_cols)} features")

    return np.array(X), np.array(y), sample_ids


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train elastic net sex classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--output",
        default="models/",
        help="Output directory for trained model (default: models/)",
    )
    parser.add_argument(
        "--name",
        default="sex_classifier",
        help="Model name (default: sex_classifier)",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Model version (semantic, default: 1.0.0)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Cross-validation folds (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Regularization strength (default: 0.5)",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.5,
        help="L1/L2 ratio for elastic net (default: 0.5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--description",
        help="Optional model description",
    )

    args = parser.parse_args()

    try:
        # Load data
        logger.info("=" * 60)
        logger.info("Training Sex Classifier Model")
        logger.info("=" * 60)

        X, y, sample_ids = load_training_data(args.data)

        # Create model config
        config = ModelConfig(
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
        )

        logger.info("\nModel Configuration:")
        logger.info(f"  Regularization strength (alpha): {config.alpha}")
        logger.info(f"  L1/L2 ratio (l1_ratio): {config.l1_ratio}")
        logger.info(f"  Cross-validation folds: {config.cv_folds}")
        logger.info(f"  Random state: {config.random_state}")

        # Train model
        logger.info("\nTraining model...")
        trainer = SexClassifierTrainer(config=config)
        results = trainer.train(X, y)

        logger.info("\nTraining Results:")
        logger.info(f"  Accuracy: {results.accuracy:.4f}")
        logger.info(f"  Precision: {results.precision:.4f}")
        logger.info(f"  Recall: {results.recall:.4f}")
        logger.info(f"  F1 Score: {results.f1_score:.4f}")
        logger.info(f"  AUC Score: {results.auc_score:.4f}")
        logger.info(f"  Mean CV Score: {results.mean_cv_score:.4f} ± {results.std_cv_score:.4f}")

        # Save model
        logger.info("\nSaving model...")
        persistence = ModelPersistence()
        output_path = Path(args.output)

        persistence.save_model(
            trainer,
            output_path,
            name=args.name,
            version=args.version,
            description=args.description,
            training_samples=len(X),
            accuracy=results.accuracy,
        )

        logger.info(f"Model saved to {output_path / args.name / args.version}")

        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Training complete: {args.name} v{args.version}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
