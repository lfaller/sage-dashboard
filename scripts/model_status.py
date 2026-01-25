#!/usr/bin/env python3
"""Check status of trained sex classifier models.

Lists available models, versions, and metadata.

Usage:
    poetry run python scripts/model_status.py
    poetry run python scripts/model_status.py --model sex_classifier
    poetry run python scripts/model_status.py --models-path /custom/path
"""

import argparse
import sys
from pathlib import Path

from src.sage.logging_config import get_logger
from src.sage.model_persistence import ModelPersistence

logger = get_logger(__name__)


def list_all_models(models_path: Path):
    """List all available models and versions."""
    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_path}")
        print(f"No models found at {models_path}")
        return

    persistence = ModelPersistence()

    model_dirs = [d for d in models_path.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"No models found in {models_path}")
        return

    print(f"\nAvailable Models ({models_path}):")
    print("=" * 80)

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        versions = persistence.list_versions(models_path, model_name)

        if not versions:
            continue

        print(f"\n{model_name}:")
        print(f"  {'Version':<12} {'Training Date':<15} {'Accuracy':<10} {'Samples':<8}")
        print(f"  {'-' * 50}")

        for version in versions:
            metadata = persistence.get_model_metadata(models_path, model_name, version)

            if metadata:
                accuracy = f"{metadata.accuracy:.4f}" if metadata.accuracy else "N/A"
                print(
                    f"  {version:<12} {metadata.training_date:<15} "
                    f"{accuracy:<10} {metadata.training_samples:<8}"
                )
                if metadata.description:
                    print(f"    Description: {metadata.description}")
            else:
                print(f"  {version:<12} {'N/A':<15} {'N/A':<10} {'N/A':<8}")


def show_model_details(models_path: Path, model_name: str):
    """Show details for a specific model."""
    persistence = ModelPersistence()

    versions = persistence.list_versions(models_path, model_name)

    if not versions:
        logger.warning(f"Model not found: {model_name}")
        print(f"Model '{model_name}' not found in {models_path}")
        return

    print(f"\nModel: {model_name}")
    print("=" * 80)

    for version in versions:
        metadata = persistence.get_model_metadata(models_path, model_name, version)

        print(f"\nVersion {version}:")
        print("-" * 40)

        if metadata:
            print(f"  Model Type: {metadata.model_type}")
            print(f"  Training Date: {metadata.training_date}")
            print(f"  Accuracy: {metadata.accuracy:.4f if metadata.accuracy else 'N/A'}")
            print(f"  Training Samples: {metadata.training_samples}")
            if metadata.description:
                print(f"  Description: {metadata.description}")

            version_dir = models_path / model_name / version
            print(f"  Path: {version_dir}")
        else:
            print("  (No metadata found)")


def main():
    """Main model status script."""
    parser = argparse.ArgumentParser(
        description="Check status of trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        help="Show details for specific model",
    )
    parser.add_argument(
        "--models-path",
        default="models/",
        help="Path to models directory (default: models/)",
    )

    args = parser.parse_args()

    try:
        models_path = Path(args.models_path)

        if args.model:
            show_model_details(models_path, args.model)
        else:
            list_all_models(models_path)

        return 0

    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
