"""Model database integration and management.

Provides ModelDatabaseManager for syncing filesystem-based models with database,
managing versions, recording validation results, and maintaining audit trails.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sage.database import get_supabase_client
from sage.model_persistence import ModelPersistence, ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class ModelVersionRecord:
    """Record of a model version in the database."""

    name: str
    """Model name."""

    version: str
    """Semantic version (e.g., 1.0.0)."""

    model_type: str
    """Type of model (e.g., elastic_net_logistic_regression)."""

    training_date: str
    """Date model was trained (ISO format)."""

    training_samples: Optional[int] = None
    """Number of samples used for training."""

    accuracy: Optional[float] = None
    """Training accuracy."""

    precision: Optional[float] = None
    """Training precision."""

    recall: Optional[float] = None
    """Training recall."""

    f1_score: Optional[float] = None
    """Training F1 score."""

    auc_score: Optional[float] = None
    """Training AUC score."""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)


class ModelDatabaseManager:
    """Manages model versions and metadata in database."""

    def __init__(self):
        """Initialize database manager."""
        self.db = self._connect_database()
        self.persistence = ModelPersistence()
        logger.info("ModelDatabaseManager initialized")

    def _connect_database(self):
        """Connect to database.

        Returns:
            Supabase client
        """
        try:
            return get_supabase_client()
        except Exception as e:
            logger.warning(f"Failed to connect to database: {e}")
            return None

    def sync_filesystem_to_db(self, model_path: str) -> None:
        """Sync filesystem models to database.

        Discovers all models in filesystem and creates database records.

        Args:
            model_path: Path to models directory
        """
        if not self.db:
            logger.warning("Database not connected, skipping sync")
            return

        model_path = Path(model_path)

        try:
            # Iterate through model directories
            for model_dir in model_path.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name

                # Iterate through versions
                for version_dir in model_dir.iterdir():
                    if not version_dir.is_dir():
                        continue

                    version = version_dir.name
                    metadata_path = version_dir / "metadata.json"

                    if metadata_path.exists():
                        try:
                            metadata = ModelMetadata.from_json_file(str(metadata_path))
                            self._insert_model_version(
                                name=model_name,
                                version=version,
                                model_type=metadata.model_type,
                                training_date=metadata.training_date,
                                training_samples=metadata.training_samples,
                                accuracy=metadata.accuracy,
                            )
                            logger.info(f"Synced model {model_name}:{version}")
                        except Exception as e:
                            logger.warning(f"Failed to sync {model_name}:{version}: {e}")

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise

    def register_model_version(
        self,
        name: str,
        version: str,
        model_type: str,
        training_date: str,
        training_samples: Optional[int] = None,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        auc_score: Optional[float] = None,
    ) -> None:
        """Register trained model version in database.

        Args:
            name: Model name
            version: Semantic version
            model_type: Type of model
            training_date: Training date (ISO format)
            training_samples: Number of training samples
            accuracy: Training accuracy
            precision: Training precision
            recall: Training recall
            f1_score: Training F1 score
            auc_score: Training AUC score
        """
        if not self.db:
            logger.warning("Database not connected, skipping registration")
            return

        record = ModelVersionRecord(
            name=name,
            version=version,
            model_type=model_type,
            training_date=training_date,
            training_samples=training_samples,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_score=auc_score,
        )

        self._insert_model_version(**record.to_dict())

    def record_validation(
        self,
        model_name: str,
        model_version: str,
        validation_date: str,
        test_samples: int,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        auc_score: float,
        sensitivity: Optional[float] = None,
        specificity: Optional[float] = None,
        true_positives: Optional[int] = None,
        true_negatives: Optional[int] = None,
        false_positives: Optional[int] = None,
        false_negatives: Optional[int] = None,
        benchmark_name: Optional[str] = None,
        benchmark_accuracy: Optional[float] = None,
    ) -> None:
        """Record validation results in database.

        Args:
            model_name: Model name
            model_version: Model version
            validation_date: Date of validation
            test_samples: Number of test samples
            accuracy: Validation accuracy
            precision: Validation precision
            recall: Validation recall
            f1_score: Validation F1 score
            auc_score: Validation AUC score
            sensitivity: Validation sensitivity
            specificity: Validation specificity
            true_positives: True positives from confusion matrix
            true_negatives: True negatives from confusion matrix
            false_positives: False positives from confusion matrix
            false_negatives: False negatives from confusion matrix
            benchmark_name: Name of benchmark if compared
            benchmark_accuracy: Benchmark accuracy if compared
        """
        if not self.db:
            logger.warning("Database not connected, skipping validation record")
            return

        self._insert_validation_result(
            model_name=model_name,
            model_version=model_version,
            validation_date=validation_date,
            test_samples=test_samples,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_score=auc_score,
            sensitivity=sensitivity,
            specificity=specificity,
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            benchmark_name=benchmark_name,
            benchmark_accuracy=benchmark_accuracy,
        )

    def get_model_versions(self, model_name: str) -> List[Dict]:
        """Get all versions of a model.

        Args:
            model_name: Model name

        Returns:
            List of version records
        """
        if not self.db:
            logger.warning("Database not connected")
            return []

        return self._query_model_versions(model_name)

    def get_active_model(self, model_name: str) -> Optional[Dict]:
        """Get active version of a model.

        Args:
            model_name: Model name

        Returns:
            Active model version record or None
        """
        if not self.db:
            logger.warning("Database not connected")
            return None

        return self._query_active_model(model_name)

    def set_active_version(self, model_name: str, version: str) -> None:
        """Set active version for a model.

        Args:
            model_name: Model name
            version: Version to set as active
        """
        if not self.db:
            logger.warning("Database not connected, skipping active version update")
            return

        self._update_active_version(model_name, version)

    def get_audit_trail(self, model_name: str) -> List[Dict]:
        """Get full audit trail for model.

        Args:
            model_name: Model name

        Returns:
            List of audit events
        """
        if not self.db:
            logger.warning("Database not connected")
            return []

        return self._query_audit_trail(model_name)

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version of a model.

        Args:
            model_name: Model name

        Returns:
            Latest version string or None
        """
        if not self.db:
            logger.warning("Database not connected")
            return None

        return self._query_latest_version(model_name)

    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a model version from database.

        Args:
            model_name: Model name
            version: Version to delete
        """
        if not self.db:
            logger.warning("Database not connected, skipping deletion")
            return

        self._delete_model_version(model_name, version)

    def get_best_performing_model(self, model_name: str) -> Optional[Dict]:
        """Get best performing version of a model.

        Args:
            model_name: Model name

        Returns:
            Best performing version record or None
        """
        if not self.db:
            logger.warning("Database not connected")
            return None

        return self._query_best_performing_model(model_name)

    def get_validation_history(self, model_name: str, version: str) -> List[Dict]:
        """Get validation history for a model version.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            List of validation records
        """
        if not self.db:
            logger.warning("Database not connected")
            return []

        return self._query_validation_history(model_name, version)

    def health_check(self) -> bool:
        """Check database connectivity and health.

        Returns:
            True if database is healthy, False otherwise
        """
        if not self.db:
            return False

        return self._test_connection()

    def export_model_report(self, model_name: str) -> Optional[Dict]:
        """Export comprehensive report for a model.

        Args:
            model_name: Model name

        Returns:
            Report dictionary with versions, metrics, and analysis
        """
        if not self.db:
            logger.warning("Database not connected")
            return None

        return self._generate_report(model_name)

    def compare_models(self) -> List[Dict]:
        """Compare performance across all models.

        Returns:
            List of model comparison records
        """
        if not self.db:
            logger.warning("Database not connected")
            return []

        return self._query_model_comparison()

    # Internal database operations (implementation details)

    def _insert_model_version(self, **kwargs) -> None:
        """Insert model version record into database."""
        try:
            self.db.table("model_versions").insert(kwargs).execute()
            logger.debug(f"Inserted model version: {kwargs.get('name')}:{kwargs.get('version')}")
        except Exception as e:
            logger.warning(f"Failed to insert model version: {e}")

    def _insert_validation_result(self, **kwargs) -> None:
        """Insert validation result into database."""
        try:
            self.db.table("validation_results").insert(kwargs).execute()
            logger.debug(f"Inserted validation result for {kwargs.get('model_name')}")
        except Exception as e:
            logger.warning(f"Failed to insert validation result: {e}")

    def _query_model_versions(self, model_name: str) -> List[Dict]:
        """Query model versions from database."""
        try:
            response = self.db.table("model_versions").select("*").eq("name", model_name).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.warning(f"Failed to query model versions: {e}")
            return []

    def _query_active_model(self, model_name: str) -> Optional[Dict]:
        """Query active model version from database."""
        try:
            response = (
                self.db.table("model_registry")
                .select("active_version")
                .eq("model_name", model_name)
                .execute()
            )
            if response.data and len(response.data) > 0:
                active_version = response.data[0].get("active_version")
                # Get full record for active version
                version_response = (
                    self.db.table("model_versions")
                    .select("*")
                    .eq("name", model_name)
                    .eq("version", active_version)
                    .execute()
                )
                return version_response.data[0] if version_response.data else None
            return None
        except Exception as e:
            logger.warning(f"Failed to query active model: {e}")
            return None

    def _update_active_version(self, model_name: str, version: str) -> None:
        """Update active version in database."""
        try:
            self.db.table("model_registry").upsert(
                {
                    "model_name": model_name,
                    "active_version": version,
                    "updated_at": datetime.now().isoformat(),
                }
            ).execute()
            logger.debug(f"Updated active version: {model_name}:{version}")
        except Exception as e:
            logger.warning(f"Failed to update active version: {e}")

    def _query_audit_trail(self, model_name: str) -> List[Dict]:
        """Query audit trail for model."""
        try:
            # Query both training and validation events
            training = (
                self.db.table("model_versions")
                .select("*")
                .eq("name", model_name)
                .order("training_date", desc=True)
                .execute()
            )

            validation = (
                self.db.table("validation_results")
                .select("*")
                .eq("model_name", model_name)
                .order("validation_date", desc=True)
                .execute()
            )

            # Combine and sort
            trail = []
            if training.data:
                trail.extend([{"event": "training", **r} for r in training.data])
            if validation.data:
                trail.extend([{"event": "validation", **r} for r in validation.data])

            return sorted(
                trail,
                key=lambda x: x.get("training_date") or x.get("validation_date"),
                reverse=True,
            )
        except Exception as e:
            logger.warning(f"Failed to query audit trail: {e}")
            return []

    def _query_latest_version(self, model_name: str) -> Optional[str]:
        """Query latest version of model."""
        try:
            response = (
                self.db.table("model_versions")
                .select("version")
                .eq("name", model_name)
                .order("training_date", desc=True)
                .limit(1)
                .execute()
            )
            return response.data[0]["version"] if response.data else None
        except Exception as e:
            logger.warning(f"Failed to query latest version: {e}")
            return None

    def _delete_model_version(self, model_name: str, version: str) -> None:
        """Delete model version from database."""
        try:
            self.db.table("model_versions").delete().eq("name", model_name).eq(
                "version", version
            ).execute()
            logger.debug(f"Deleted model version: {model_name}:{version}")
        except Exception as e:
            logger.warning(f"Failed to delete model version: {e}")

    def _query_best_performing_model(self, model_name: str) -> Optional[Dict]:
        """Query best performing version."""
        try:
            response = (
                self.db.table("model_versions")
                .select("*")
                .eq("name", model_name)
                .order("accuracy", desc=True)
                .limit(1)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.warning(f"Failed to query best performing model: {e}")
            return None

    def _query_validation_history(self, model_name: str, version: str) -> List[Dict]:
        """Query validation history for model version."""
        try:
            response = (
                self.db.table("validation_results")
                .select("*")
                .eq("model_name", model_name)
                .eq("model_version", version)
                .order("validation_date", desc=True)
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            logger.warning(f"Failed to query validation history: {e}")
            return []

    def _test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            self.db.table("model_versions").select("*").limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    def _generate_report(self, model_name: str) -> Optional[Dict]:
        """Generate comprehensive model report."""
        try:
            versions = self._query_model_versions(model_name)
            best = self._query_best_performing_model(model_name)
            audit = self._query_audit_trail(model_name)

            return {
                "name": model_name,
                "versions": versions,
                "best_version": best.get("version") if best else None,
                "best_accuracy": best.get("accuracy") if best else None,
                "audit_trail": audit,
            }
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")
            return None

    def _query_model_comparison(self) -> List[Dict]:
        """Query comparison across all models."""
        try:
            # Get all unique model names
            response = self.db.table("model_versions").select("name").execute()

            if not response.data:
                return []

            model_names = list(set(r["name"] for r in response.data))
            comparison = []

            for name in model_names:
                best = self._query_best_performing_model(name)
                if best:
                    comparison.append(
                        {
                            "name": name,
                            "best_accuracy": best.get("accuracy"),
                            "best_version": best.get("version"),
                        }
                    )

            return sorted(comparison, key=lambda x: x.get("best_accuracy", 0), reverse=True)
        except Exception as e:
            logger.warning(f"Failed to query model comparison: {e}")
            return []
