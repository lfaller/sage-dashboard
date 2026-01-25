"""Tests for model database integration and management.

Tests ModelDatabaseManager for syncing filesystem models to database,
recording validation results, and managing model versions.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.sage.model_database import ModelDatabaseManager, ModelVersionRecord
from src.sage.model_training import SexClassifierTrainer
from src.sage.model_persistence import ModelPersistence


class TestModelVersionRecord:
    """Test ModelVersionRecord dataclass."""

    def test_initialization(self):
        """Should initialize with required fields."""
        record = ModelVersionRecord(
            name="sex_classifier",
            version="1.0.0",
            model_type="elastic_net_logistic_regression",
            training_date="2026-01-25",
            training_samples=100,
            accuracy=0.92,
        )

        assert record.name == "sex_classifier"
        assert record.version == "1.0.0"
        assert record.accuracy == 0.92

    def test_to_dict_serialization(self):
        """Should serialize to dictionary."""
        record = ModelVersionRecord(
            name="sex_classifier",
            version="1.0.0",
            model_type="elastic_net_logistic_regression",
            training_date="2026-01-25",
            training_samples=100,
            accuracy=0.92,
        )

        data = record.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "sex_classifier"
        assert data["version"] == "1.0.0"


class TestModelDatabaseManager:
    """Test ModelDatabaseManager for database integration."""

    def test_initialization(self):
        """Should initialize database manager."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            manager = ModelDatabaseManager()
            assert manager is not None

    def test_sync_filesystem_to_db_creates_records(self):
        """Should sync filesystem models to database."""
        trainer = SexClassifierTrainer()
        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)
        trainer.train(X, y)

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

            # Mock database manager
            with patch.object(ModelDatabaseManager, "_connect_database"):
                with patch.object(ModelDatabaseManager, "_insert_model_version") as mock_insert:
                    manager = ModelDatabaseManager()
                    manager.sync_filesystem_to_db(str(model_path))

                    # Verify insert was called
                    assert mock_insert.called

    def test_register_model_version(self):
        """Should register model version in database."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_insert_model_version") as mock_insert:
                manager = ModelDatabaseManager()

                manager.register_model_version(
                    name="test_model",
                    version="1.0.0",
                    model_type="elastic_net_logistic_regression",
                    training_date="2026-01-25",
                    training_samples=100,
                    accuracy=0.92,
                )

                assert mock_insert.called

    def test_record_validation_result(self):
        """Should record validation result in database."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_insert_validation_result") as mock_insert:
                manager = ModelDatabaseManager()

                manager.record_validation(
                    model_name="test_model",
                    model_version="1.0.0",
                    validation_date="2026-01-25",
                    test_samples=20,
                    accuracy=0.90,
                    precision=0.91,
                    recall=0.89,
                    f1_score=0.90,
                    auc_score=0.95,
                )

                assert mock_insert.called

    def test_get_model_versions(self):
        """Should retrieve model versions from database."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_query_model_versions") as mock_query:
                mock_query.return_value = [
                    {"name": "test", "version": "1.0.0", "accuracy": 0.92},
                    {"name": "test", "version": "2.0.0", "accuracy": 0.94},
                ]

                manager = ModelDatabaseManager()
                versions = manager.get_model_versions("test")

                assert len(versions) == 2
                assert versions[0]["version"] == "1.0.0"

    def test_get_active_model(self):
        """Should retrieve active model version."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_query_active_model") as mock_query:
                mock_query.return_value = {"version": "2.0.0", "accuracy": 0.94}

                manager = ModelDatabaseManager()
                active = manager.get_active_model("test")

                assert active["version"] == "2.0.0"

    def test_set_active_version(self):
        """Should set active model version."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_update_active_version") as mock_update:
                manager = ModelDatabaseManager()

                manager.set_active_version("test", "2.0.0")

                assert mock_update.called

    def test_get_audit_trail(self):
        """Should retrieve audit trail for model."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_query_audit_trail") as mock_query:
                mock_query.return_value = [
                    {
                        "event": "training",
                        "timestamp": "2026-01-20",
                        "version": "1.0.0",
                        "accuracy": 0.92,
                    },
                    {
                        "event": "validation",
                        "timestamp": "2026-01-21",
                        "version": "1.0.0",
                        "accuracy": 0.91,
                    },
                ]

                manager = ModelDatabaseManager()
                trail = manager.get_audit_trail("test")

                assert len(trail) == 2
                assert trail[0]["event"] == "training"

    def test_get_latest_version(self):
        """Should return latest version for model."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_query_latest_version") as mock_query:
                mock_query.return_value = "2.0.0"

                manager = ModelDatabaseManager()
                latest = manager.get_latest_version("test")

                assert latest == "2.0.0"

    def test_delete_model_version(self):
        """Should delete model version from database."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_delete_model_version") as mock_delete:
                manager = ModelDatabaseManager()

                manager.delete_model_version("test", "1.0.0")

                assert mock_delete.called

    def test_get_best_performing_model(self):
        """Should return best performing model across versions."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_query_best_performing_model") as mock_query:
                mock_query.return_value = {"version": "2.0.0", "accuracy": 0.95}

                manager = ModelDatabaseManager()
                best = manager.get_best_performing_model("test")

                assert best["version"] == "2.0.0"
                assert best["accuracy"] == 0.95

    def test_get_validation_history(self):
        """Should retrieve validation history for model version."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_query_validation_history") as mock_query:
                mock_query.return_value = [
                    {"validation_date": "2026-01-21", "accuracy": 0.90},
                    {"validation_date": "2026-01-22", "accuracy": 0.91},
                ]

                manager = ModelDatabaseManager()
                history = manager.get_validation_history("test", "1.0.0")

                assert len(history) == 2
                assert history[0]["accuracy"] == 0.90

    def test_database_health_check(self):
        """Should verify database connectivity."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_test_connection") as mock_test:
                mock_test.return_value = True

                manager = ModelDatabaseManager()
                healthy = manager.health_check()

                assert healthy is True

    def test_export_model_report(self):
        """Should export comprehensive model report."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_generate_report") as mock_report:
                mock_report.return_value = {
                    "name": "test",
                    "versions": [
                        {"version": "1.0.0", "accuracy": 0.92},
                        {"version": "2.0.0", "accuracy": 0.94},
                    ],
                    "best_version": "2.0.0",
                }

                manager = ModelDatabaseManager()
                report = manager.export_model_report("test")

                assert report["name"] == "test"
                assert report["best_version"] == "2.0.0"

    def test_comparison_across_models(self):
        """Should compare performance across different models."""
        with patch.object(ModelDatabaseManager, "_connect_database"):
            with patch.object(ModelDatabaseManager, "_query_model_comparison") as mock_query:
                mock_query.return_value = [
                    {"name": "model_a", "best_accuracy": 0.92},
                    {"name": "model_b", "best_accuracy": 0.94},
                ]

                manager = ModelDatabaseManager()
                comparison = manager.compare_models()

                assert len(comparison) == 2
                assert comparison[1]["best_accuracy"] == 0.94
