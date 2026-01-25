"""Model training and orchestration for sex classification.

Trains elastic net logistic regression models on expression data
for sex classification with cross-validation and checkpointing support.

Based on Flynn et al. (2021) BMC Bioinformatics methodology.
"""

import logging
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for elastic net sex classifier training."""

    alpha: float = 0.5
    """L1/L2 ratio for elastic net (0.5 = balanced)."""

    l1_ratio: float = 0.5
    """Regularization strength (0-1, higher = more regularization)."""

    random_state: int = 42
    """Random seed for reproducibility."""

    cv_folds: int = 10
    """Number of cross-validation folds."""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class TrainingResults:
    """Results from model training and cross-validation."""

    accuracy: float
    """Overall accuracy across all CV folds."""

    precision: float
    """Weighted precision across folds."""

    recall: float
    """Weighted recall across folds."""

    f1_score: float
    """Weighted F1 score across folds."""

    auc_score: float
    """Area under ROC curve across folds."""

    cv_fold_scores: np.ndarray
    """Individual CV fold scores."""

    @property
    def mean_cv_score(self) -> float:
        """Mean cross-validation score."""
        return float(np.mean(self.cv_fold_scores))

    @property
    def std_cv_score(self) -> float:
        """Standard deviation of CV scores."""
        return float(np.std(self.cv_fold_scores))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        data["cv_fold_scores"] = self.cv_fold_scores.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingResults":
        """Deserialize from dictionary."""
        data_copy = data.copy()
        data_copy["cv_fold_scores"] = np.array(data_copy["cv_fold_scores"])
        return cls(**data_copy)


class SexClassifierTrainer:
    """Trains elastic net sex classifier with cross-validation."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model: Optional[Any] = None,
    ):
        """Initialize trainer.

        Args:
            config: ModelConfig with hyperparameters
            model: Pre-trained classifier model (optional)
        """
        self.config = config or ModelConfig()
        self.trainer_model = model
        self.results: Optional[TrainingResults] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TrainingResults:
        """Train elastic net classifier with cross-validation.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - 0=female, 1=male

        Returns:
            TrainingResults with cross-validation scores
        """
        logger.info(
            f"Training elastic net with alpha={self.config.alpha}, "
            f"l1_ratio={self.config.l1_ratio}, cv_folds={self.config.cv_folds}"
        )

        # Create elastic net classifier using SGDClassifier
        # (supports full elastic net with alpha and l1_ratio parameters)
        model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            random_state=self.config.random_state,
            max_iter=1000,
            eta0=0.01,
        )

        # Define scoring metrics
        scoring = {
            "accuracy": "accuracy",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "f1_weighted": "f1_weighted",
            "roc_auc": "roc_auc",
        }

        # Stratified k-fold cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Run cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

        # Train final model on all data with calibration for predict_proba
        # Create calibrated classifier that wraps SGDClassifier
        base_model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            random_state=self.config.random_state,
            max_iter=1000,
            eta0=0.01,
        )

        # Wrap with calibration to add predict_proba method
        # Use cv=5 for calibration (internal cross-validation for probability mapping)
        self.trainer_model = CalibratedClassifierCV(base_model, cv=5)
        self.trainer_model.fit(X, y)

        # Extract metrics
        results = TrainingResults(
            accuracy=float(np.mean(cv_results["test_accuracy"])),
            precision=float(np.mean(cv_results["test_precision_weighted"])),
            recall=float(np.mean(cv_results["test_recall_weighted"])),
            f1_score=float(np.mean(cv_results["test_f1_weighted"])),
            auc_score=float(np.mean(cv_results["test_roc_auc"])),
            cv_fold_scores=cv_results["test_accuracy"],
        )

        self.results = results

        logger.info(
            f"Training complete: accuracy={results.accuracy:.4f}, "
            f"f1={results.f1_score:.4f}, auc={results.auc_score:.4f}"
        )

        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 2) - P(female), P(male)

        Raises:
            ValueError: If model not trained yet
        """
        if self.trainer_model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.trainer_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict sex labels for samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,) - 0=female, 1=male

        Raises:
            ValueError: If model not trained yet
        """
        if self.trainer_model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.trainer_model.predict(X)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to file.

        Args:
            path: Path to save checkpoint
        """
        if self.trainer_model is None:
            raise ValueError("No model to save. Train first.")

        checkpoint = {
            "model": self.trainer_model,
            "config": self.config,
            "results": self.results.to_dict() if self.results else None,
        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint from file.

        Args:
            path: Path to checkpoint file
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.trainer_model = checkpoint["model"]
        self.config = checkpoint["config"]

        if checkpoint["results"]:
            self.results = TrainingResults.from_dict(checkpoint["results"])

        logger.info(f"Model checkpoint loaded from {path}")

    def get_feature_importance(self) -> np.ndarray:
        """Get feature coefficients from trained model.

        Returns:
            Feature coefficients (n_features,)

        Raises:
            ValueError: If model not trained yet
        """
        if self.trainer_model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Get coefficients for positive class (male)
        # Note: trainer_model is a CalibratedClassifierCV wrapper around SGDClassifier
        # Access the underlying estimator to get coef_
        base_model = self.trainer_model.estimators_[0]
        return base_model.coef_[0]
