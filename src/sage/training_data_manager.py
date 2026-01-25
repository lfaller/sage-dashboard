"""Training data extraction and management for sex classification models.

Manages extraction of high-confidence sex labels from SAGE studies for training
elastic net logistic regression models. Handles label conflict resolution, dataset
statistics, and stratified train/test splitting for balanced validation.

Based on methodology from Flynn et al. (2021) BMC Bioinformatics.
"""

import json
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class TrainingDatasetMetadata:
    """Metadata for training dataset fixtures."""

    name: str
    version: str
    total_samples: int
    male_count: int
    female_count: int
    description: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingDatasetMetadata":
        """Deserialize metadata from dictionary."""
        return cls(**data)


@dataclass
class TrainingDataset:
    """Dataset of training samples with sex labels."""

    samples: List[str]
    labels: np.ndarray

    @property
    def male_count(self) -> int:
        """Count of male samples (label=1)."""
        return int(np.sum(self.labels == 1))

    @property
    def female_count(self) -> int:
        """Count of female samples (label=0)."""
        return int(np.sum(self.labels == 0))

    @property
    def total_samples(self) -> int:
        """Total number of samples."""
        return len(self.labels)

    @property
    def balance_ratio(self) -> float:
        """Ratio of male samples (male / total)."""
        if self.total_samples == 0:
            return 0.0
        return self.male_count / self.total_samples

    def to_labels_array(self) -> np.ndarray:
        """Return labels as numpy array."""
        return self.labels

    def stratified_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple["TrainingDataset", "TrainingDataset"]:
        """Split dataset into train/test with stratification.

        Maintains class balance ratio in both splits.

        Args:
            test_size: Proportion of data for test set (default: 0.2)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )

        train_idx, test_idx = next(splitter.split(self.samples, self.labels))

        train_samples = [self.samples[i] for i in train_idx]
        train_labels = self.labels[train_idx]

        test_samples = [self.samples[i] for i in test_idx]
        test_labels = self.labels[test_idx]

        return (
            TrainingDataset(samples=train_samples, labels=train_labels),
            TrainingDataset(samples=test_samples, labels=test_labels),
        )


class TrainingDataExtractor:
    """Extracts training data from studies with high-confidence sex labels."""

    def __init__(self):
        """Initialize TrainingDataExtractor."""
        logger.debug("TrainingDataExtractor initialized")

    def fetch_from_database(self) -> List[Dict]:
        """Fetch high-confidence samples from database.

        This is a placeholder that should be implemented to query
        the actual training database when ready.

        Returns:
            List of sample dictionaries with sex and confidence metadata
        """
        return []

    def fetch_high_confidence_samples(self, threshold: float = 0.90) -> List[Dict]:
        """Fetch samples with confidence >= threshold.

        Args:
            threshold: Minimum confidence score (0.0-1.0)

        Returns:
            List of samples with metadata: gsm_id, sex, confidence, source, etc.
        """
        samples = self.fetch_from_database()
        return [s for s in samples if s.get("confidence", 0) >= threshold]

    def validate_label_consistency(self, samples: List[Dict]) -> List[Dict]:
        """Detect conflicting sex labels for same sample.

        Args:
            samples: List of samples potentially with duplicates

        Returns:
            List of conflicts found (empty if all consistent)
        """
        # Group by sample ID
        sample_groups = {}
        for sample in samples:
            gsm_id = sample.get("gsm_id")
            if gsm_id:
                if gsm_id not in sample_groups:
                    sample_groups[gsm_id] = []
                sample_groups[gsm_id].append(sample)

        # Find conflicts
        conflicts = []
        for gsm_id, group in sample_groups.items():
            if len(group) > 1:
                # Check if all have same sex label
                sexes = {s.get("sex") for s in group}
                if len(sexes) > 1:
                    conflicts.append(
                        {
                            "gsm_id": gsm_id,
                            "conflicting_labels": list(sexes),
                            "sources": [s.get("source") for s in group],
                        }
                    )

        return conflicts

    def resolve_conflicting_labels(
        self, samples: List[Dict], preference_order: Optional[List[str]] = None
    ) -> List[Dict]:
        """Resolve conflicting labels by preference order.

        Default preference: characteristics > sample_names > other

        Args:
            samples: Samples potentially with conflicts
            preference_order: Preferred source order (default: characteristics first)

        Returns:
            Deduplicated samples with preferred label kept
        """
        if preference_order is None:
            preference_order = ["characteristics", "sample_names"]

        # Group by sample ID
        sample_groups = {}
        for sample in samples:
            gsm_id = sample.get("gsm_id")
            if gsm_id:
                if gsm_id not in sample_groups:
                    sample_groups[gsm_id] = []
                sample_groups[gsm_id].append(sample)

        # Resolve conflicts using preference order
        resolved = []
        for gsm_id, group in sample_groups.items():
            if len(group) == 1:
                resolved.append(group[0])
            else:
                # Find highest preference source
                selected = group[0]
                selected_pref = float("inf")

                for sample in group:
                    source = sample.get("source", "")
                    try:
                        pref = preference_order.index(source)
                        if pref < selected_pref:
                            selected = sample
                            selected_pref = pref
                    except ValueError:
                        # Source not in preference list, skip
                        pass

                resolved.append(selected)

        return resolved

    def export_training_fixture(
        self,
        training_data: List[Dict],
        output_path: str,
        version: str = "1.0.0",
        name: str = "sage_training",
    ) -> None:
        """Export training data as JSON fixture.

        Args:
            training_data: List of training samples with metadata
            output_path: Path to write JSON fixture
            version: Fixture version (default: 1.0.0)
            name: Fixture name (default: sage_training)
        """
        # Build metadata
        male_count = sum(1 for s in training_data if s.get("sex") == "male")
        female_count = sum(1 for s in training_data if s.get("sex") == "female")

        metadata = {
            "name": name,
            "version": version,
            "total_samples": len(training_data),
            "male_count": male_count,
            "female_count": female_count,
        }

        # Build fixture
        fixture = {"metadata": metadata, "samples": training_data}

        # Write to file
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(fixture, f, indent=2)

        logger.info(f"Exported training fixture: {output_path}")

    def compute_dataset_statistics(self, training_data: List[Dict]) -> Dict:
        """Compute statistics on training dataset.

        Args:
            training_data: List of training samples

        Returns:
            Dictionary with: total_samples, male_count, female_count, balance_ratio
        """
        total = len(training_data)
        male_count = sum(1 for s in training_data if s.get("sex") == "male")
        female_count = sum(1 for s in training_data if s.get("sex") == "female")

        balance_ratio = male_count / total if total > 0 else 0.0

        return {
            "total_samples": total,
            "male_count": male_count,
            "female_count": female_count,
            "balance_ratio": balance_ratio,
        }

    def validate_training_set_size(self, training_data: List[Dict]) -> None:
        """Validate training set has sufficient samples.

        Issues warning if dataset has fewer than 20 samples.

        Args:
            training_data: List of training samples
        """
        if len(training_data) < 20:
            warnings.warn(
                f"Training set has fewer than 20 samples ({len(training_data)}). "
                "This may lead to overfitting. Consider collecting more data.",
                UserWarning,
            )
