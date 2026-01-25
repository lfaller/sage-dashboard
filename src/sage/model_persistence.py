"""Model persistence and version management for sex classifiers.

Handles saving, loading, and managing trained sex classifier models
with semantic versioning and model registry.

Based on Flynn et al. (2021) BMC Bioinformatics methodology.
"""

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a saved model."""

    name: str
    """Model name/identifier."""

    version: str
    """Semantic version (e.g., 1.0.0)."""

    model_type: str
    """Type of model (e.g., elastic_net_logistic_regression)."""

    training_date: str
    """Date model was trained (YYYY-MM-DD)."""

    description: Optional[str] = None
    """Optional description of model."""

    training_samples: int = 0
    """Number of samples used for training."""

    accuracy: Optional[float] = None
    """Cross-validation accuracy if available."""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Deserialize from dictionary."""
        return cls(**data)


class ModelPersistence:
    """Manages saving and loading trained models with versioning."""

    def __init__(self):
        """Initialize model persistence manager."""
        logger.debug("ModelPersistence initialized")

    def save_model(
        self,
        trainer: object,
        model_path: Path,
        name: str,
        version: str,
        description: Optional[str] = None,
        training_samples: int = 0,
        accuracy: Optional[float] = None,
    ) -> None:
        """Save trained model with metadata.

        Args:
            trainer: Trained SexClassifierTrainer instance
            model_path: Base directory for model storage
            name: Model name
            version: Semantic version (e.g., 1.0.0)
            description: Optional description
            training_samples: Number of training samples
            accuracy: Cross-validation accuracy
        """
        model_path = Path(model_path)
        version_dir = model_path / name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model checkpoint
        model_file = version_dir / "model.pkl"
        trainer.save_checkpoint(str(model_file))

        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type="elastic_net_logistic_regression",
            training_date=datetime.now().strftime("%Y-%m-%d"),
            description=description,
            training_samples=training_samples,
            accuracy=accuracy,
        )

        # Save metadata
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Model saved: {name}/{version} at {version_dir}")

    def load_model(
        self,
        model_path: Path,
        name: str,
        version: str,
    ) -> object:
        """Load previously saved model.

        Args:
            model_path: Base directory for model storage
            name: Model name
            version: Semantic version

        Returns:
            Loaded SexClassifierTrainer instance
        """
        from sage.model_training import SexClassifierTrainer

        model_path = Path(model_path)
        version_dir = model_path / name / version

        if not version_dir.exists():
            raise FileNotFoundError(f"Model not found: {name}/{version}")

        model_file = version_dir / "model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load model
        trainer = SexClassifierTrainer()
        trainer.load_checkpoint(str(model_file))

        logger.info(f"Model loaded: {name}/{version} from {version_dir}")

        return trainer

    def list_versions(self, model_path: Path, name: str) -> List[str]:
        """List all versions of a model.

        Args:
            model_path: Base directory for model storage
            name: Model name

        Returns:
            List of version strings (sorted)
        """
        model_path = Path(model_path)
        model_dir = model_path / name

        if not model_dir.exists():
            return []

        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        return sorted(versions)

    def get_latest_version(self, model_path: Path, name: str) -> Optional[str]:
        """Get latest version of a model.

        Args:
            model_path: Base directory for model storage
            name: Model name

        Returns:
            Latest version string or None if no versions exist
        """
        versions = self.list_versions(model_path, name)
        if not versions:
            return None

        # Sort versions (simple string sort works for semantic versioning)
        return sorted(versions)[-1]

    def delete_model(
        self,
        model_path: Path,
        name: str,
        version: str,
    ) -> None:
        """Delete a specific model version.

        Args:
            model_path: Base directory for model storage
            name: Model name
            version: Semantic version to delete
        """
        model_path = Path(model_path)
        version_dir = model_path / name / version

        if not version_dir.exists():
            logger.warning(f"Model not found: {name}/{version}")
            return

        shutil.rmtree(version_dir)
        logger.info(f"Model deleted: {name}/{version}")

    def get_model_metadata(
        self,
        model_path: Path,
        name: str,
        version: str,
    ) -> Optional[ModelMetadata]:
        """Get metadata for a model version.

        Args:
            model_path: Base directory for model storage
            name: Model name
            version: Semantic version

        Returns:
            ModelMetadata or None if not found
        """
        model_path = Path(model_path)
        metadata_file = model_path / name / version / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            data = json.load(f)

        return ModelMetadata.from_dict(data)


class ModelRegistry:
    """Central registry for managing trained models."""

    def __init__(self):
        """Initialize model registry."""
        self._registry: Dict[str, Dict] = {}
        self._active_versions: Dict[str, str] = {}
        logger.debug("ModelRegistry initialized")

    def register(
        self,
        name: str,
        version: str,
        model_path: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Register a model in the registry.

        Args:
            name: Model name
            version: Semantic version
            model_path: Path to model storage
            description: Optional description
            metadata: Optional metadata dictionary
        """
        key = f"{name}/{version}"

        self._registry[key] = {
            "name": name,
            "version": version,
            "model_path": model_path,
            "description": description,
            "metadata": metadata or {},
            "registered_date": datetime.now().isoformat(),
        }

        logger.info(f"Model registered: {key}")

    def get(self, name: str, version: str) -> Optional[Dict]:
        """Get registered model entry.

        Args:
            name: Model name
            version: Semantic version

        Returns:
            Model entry dict or None if not found
        """
        key = f"{name}/{version}"
        return self._registry.get(key)

    def list(self) -> List[Dict]:
        """List all registered models.

        Returns:
            List of model entry dicts
        """
        return list(self._registry.values())

    def set_active(self, name: str, version: str) -> None:
        """Set a model version as active for the model name.

        Args:
            name: Model name
            version: Semantic version to activate
        """
        self._active_versions[name] = version
        logger.info(f"Active version set: {name} -> {version}")

    def get_active(self, name: str) -> Optional[str]:
        """Get active version for a model.

        Args:
            name: Model name

        Returns:
            Active version string or None
        """
        return self._active_versions.get(name)

    def delete(self, name: str, version: str) -> None:
        """Unregister a model version.

        Args:
            name: Model name
            version: Semantic version
        """
        key = f"{name}/{version}"
        if key in self._registry:
            del self._registry[key]
            logger.info(f"Model unregistered: {key}")
