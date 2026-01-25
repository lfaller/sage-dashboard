"""Tests for model persistence and version management (Phase 6A.2).

Tests ModelPersistence class for saving, loading, and version management
of trained sex classifiers.
"""

import tempfile
from pathlib import Path

import numpy as np

from src.sage.model_training import SexClassifierTrainer
from src.sage.model_persistence import ModelPersistence, ModelMetadata, ModelRegistry


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_initialization_with_required_fields(self):
        """Should initialize with required fields."""
        metadata = ModelMetadata(
            name="sex_classifier",
            version="1.0.0",
            model_type="elastic_net_logistic_regression",
            training_date="2026-01-25",
        )

        assert metadata.name == "sex_classifier"
        assert metadata.version == "1.0.0"

    def test_serialization_to_dict(self):
        """Should serialize to dictionary."""
        metadata = ModelMetadata(
            name="sex_classifier",
            version="1.0.0",
            model_type="elastic_net_logistic_regression",
            training_date="2026-01-25",
            description="Trained on microarray data",
        )

        data = metadata.to_dict()

        assert data["name"] == "sex_classifier"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Trained on microarray data"

    def test_deserialization_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "name": "sex_classifier",
            "version": "1.0.0",
            "model_type": "elastic_net_logistic_regression",
            "training_date": "2026-01-25",
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.name == "sex_classifier"
        assert metadata.version == "1.0.0"


class TestModelPersistence:
    """Test ModelPersistence for model save/load."""

    def test_initialization(self):
        """Should initialize persistence manager."""
        persistence = ModelPersistence()
        assert persistence is not None

    def test_save_model(self):
        """Should save trained model."""
        trainer = SexClassifierTrainer()
        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)
        trainer.train(X, y)

        persistence = ModelPersistence()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models"

            persistence.save_model(
                trainer,
                model_path,
                name="test_model",
                version="1.0.0",
            )

            # Check files were created
            assert (model_path / "test_model" / "1.0.0" / "model.pkl").exists()
            assert (model_path / "test_model" / "1.0.0" / "metadata.json").exists()

    def test_save_model_creates_version_directory(self):
        """Should create version-specific directory."""
        trainer = SexClassifierTrainer()
        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)
        trainer.train(X, y)

        persistence = ModelPersistence()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models"

            persistence.save_model(
                trainer,
                model_path,
                name="test_model",
                version="2.1.0",
            )

            # Check version directory exists
            version_dir = model_path / "test_model" / "2.1.0"
            assert version_dir.exists()

    def test_load_model(self):
        """Should load previously saved model."""
        trainer = SexClassifierTrainer()
        X_train = np.random.randn(40, 5)
        y_train = np.array([0] * 20 + [1] * 20)
        trainer.train(X_train, y_train)

        persistence = ModelPersistence()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models"

            # Save
            persistence.save_model(
                trainer,
                model_path,
                name="test_model",
                version="1.0.0",
            )

            # Load
            loaded_trainer = persistence.load_model(
                model_path,
                name="test_model",
                version="1.0.0",
            )

            assert loaded_trainer is not None

            # Test that loaded model can predict
            X_test = np.random.randn(10, 5)
            predictions = loaded_trainer.predict(X_test)
            assert len(predictions) == 10

    def test_model_list_versions(self):
        """Should list available versions for a model."""
        trainer = SexClassifierTrainer()
        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)
        trainer.train(X, y)

        persistence = ModelPersistence()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models"

            # Save multiple versions
            for version in ["1.0.0", "1.1.0", "2.0.0"]:
                persistence.save_model(
                    trainer,
                    model_path,
                    name="test_model",
                    version=version,
                )

            # List versions
            versions = persistence.list_versions(model_path, "test_model")

            assert len(versions) == 3
            assert "1.0.0" in versions
            assert "2.0.0" in versions

    def test_get_latest_version(self):
        """Should return latest version of model."""
        trainer = SexClassifierTrainer()
        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)
        trainer.train(X, y)

        persistence = ModelPersistence()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models"

            # Save multiple versions
            versions_saved = ["1.0.0", "1.1.0", "2.0.0"]
            for version in versions_saved:
                persistence.save_model(
                    trainer,
                    model_path,
                    name="test_model",
                    version=version,
                )

            # Get latest
            latest = persistence.get_latest_version(model_path, "test_model")

            assert latest == "2.0.0"

    def test_delete_model_version(self):
        """Should delete specific model version."""
        trainer = SexClassifierTrainer()
        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)
        trainer.train(X, y)

        persistence = ModelPersistence()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models"

            # Save two versions
            persistence.save_model(trainer, model_path, name="test", version="1.0.0")
            persistence.save_model(trainer, model_path, name="test", version="2.0.0")

            # Delete version 1.0.0
            persistence.delete_model(model_path, "test", "1.0.0")

            # Check only 2.0.0 remains
            versions = persistence.list_versions(model_path, "test")
            assert "1.0.0" not in versions
            assert "2.0.0" in versions


class TestModelRegistry:
    """Test ModelRegistry for model management."""

    def test_initialization(self):
        """Should initialize registry."""
        registry = ModelRegistry()
        assert registry is not None

    def test_register_model(self):
        """Should register model metadata."""
        trainer = SexClassifierTrainer()
        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)
        trainer.train(X, y)

        registry = ModelRegistry()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models"
            persistence = ModelPersistence()

            # Save model
            persistence.save_model(
                trainer,
                model_path,
                name="test_model",
                version="1.0.0",
            )

            # Register
            registry.register(
                name="test_model",
                version="1.0.0",
                model_path=str(model_path),
                description="Test model v1",
            )

            # Verify registration
            entry = registry.get("test_model", "1.0.0")
            assert entry is not None
            assert entry["name"] == "test_model"
            assert entry["version"] == "1.0.0"

    def test_list_registered_models(self):
        """Should list all registered models."""
        registry = ModelRegistry()

        registry.register(
            name="model1",
            version="1.0.0",
            model_path="/path/to/models",
            description="First model",
        )
        registry.register(
            name="model2",
            version="1.0.0",
            model_path="/path/to/models",
            description="Second model",
        )

        models = registry.list()

        assert len(models) >= 2

    def test_set_active_model(self):
        """Should set active model version."""
        registry = ModelRegistry()

        registry.register(
            name="test_model",
            version="1.0.0",
            model_path="/path/to/models",
        )

        registry.set_active("test_model", "1.0.0")

        active = registry.get_active("test_model")
        assert active == "1.0.0"

    def test_model_metadata_storage(self):
        """Should store and retrieve model metadata."""
        registry = ModelRegistry()

        metadata = {
            "name": "test_model",
            "version": "1.0.0",
            "trained_on": "microarray_data",
            "accuracy": 0.92,
            "training_date": "2026-01-25",
        }

        registry.register(
            name="test_model",
            version="1.0.0",
            model_path="/path/to/models",
            metadata=metadata,
        )

        entry = registry.get("test_model", "1.0.0")
        assert entry["metadata"]["accuracy"] == 0.92
