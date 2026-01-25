"""Tests for model training and orchestration (Phase 6A.2 continued).

Tests SexClassifierTrainer and model training pipeline for elastic net
sex classification model.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.sage.training_data_manager import TrainingDataset
from src.sage.model_training import (
    SexClassifierTrainer,
    ModelConfig,
    TrainingResults,
)


class TestModelConfig:
    """Test ModelConfig dataclass for hyperparameters."""

    def test_initialization_with_defaults(self):
        """Should initialize with sensible defaults."""
        config = ModelConfig()

        assert config.alpha == 0.5
        assert config.l1_ratio == 0.5
        assert config.random_state == 42
        assert config.cv_folds == 10

    def test_initialization_with_custom_values(self):
        """Should accept custom hyperparameters."""
        config = ModelConfig(
            alpha=0.3,
            l1_ratio=0.7,
            random_state=123,
            cv_folds=5,
        )

        assert config.alpha == 0.3
        assert config.l1_ratio == 0.7
        assert config.random_state == 123
        assert config.cv_folds == 5

    def test_to_dict_serialization(self):
        """Should serialize to dictionary."""
        config = ModelConfig(alpha=0.4, l1_ratio=0.6)
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["alpha"] == 0.4
        assert data["l1_ratio"] == 0.6
        assert data["random_state"] == 42

    def test_from_dict_deserialization(self):
        """Should deserialize from dictionary."""
        data = {
            "alpha": 0.3,
            "l1_ratio": 0.8,
            "random_state": 99,
            "cv_folds": 7,
        }
        config = ModelConfig.from_dict(data)

        assert config.alpha == 0.3
        assert config.l1_ratio == 0.8
        assert config.random_state == 99
        assert config.cv_folds == 7


class TestTrainingResults:
    """Test TrainingResults dataclass for training metrics."""

    def test_initialization_with_required_fields(self):
        """Should initialize with required fields."""
        results = TrainingResults(
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            auc_score=0.95,
            cv_fold_scores=np.array([0.90, 0.91, 0.92]),
        )

        assert results.accuracy == 0.92
        assert results.precision == 0.90
        assert len(results.cv_fold_scores) == 3

    def test_to_dict_serialization(self):
        """Should serialize to dictionary."""
        results = TrainingResults(
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            auc_score=0.95,
            cv_fold_scores=np.array([0.90, 0.91, 0.92]),
        )
        data = results.to_dict()

        assert isinstance(data, dict)
        assert data["accuracy"] == 0.92
        assert "cv_fold_scores" in data

    def test_mean_cv_score(self):
        """Should compute mean of cross-validation scores."""
        results = TrainingResults(
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            auc_score=0.95,
            cv_fold_scores=np.array([0.90, 0.92, 0.94]),
        )

        assert pytest.approx(results.mean_cv_score) == 0.92


class TestSexClassifierTrainerInitialization:
    """Test SexClassifierTrainer initialization."""

    def test_initialization_with_default_config(self):
        """Should initialize with default configuration."""
        trainer = SexClassifierTrainer()

        assert trainer.config is not None
        assert trainer.config.alpha == 0.5
        assert trainer.trainer_model is None
        assert trainer.results is None

    def test_initialization_with_custom_config(self):
        """Should accept custom configuration."""
        config = ModelConfig(alpha=0.3, cv_folds=5)
        trainer = SexClassifierTrainer(config=config)

        assert trainer.config.alpha == 0.3
        assert trainer.config.cv_folds == 5

    def test_initialization_with_pretrained_model(self):
        """Should accept pre-trained model."""
        mock_model = MagicMock()
        trainer = SexClassifierTrainer(model=mock_model)

        assert trainer.trainer_model is mock_model


class TestSexClassifierTrainerTraining:
    """Test model training functionality."""

    def test_train_balanced_dataset(self):
        """Should train model successfully on balanced dataset."""
        trainer = SexClassifierTrainer()

        # Create balanced training data
        X = np.random.randn(40, 5)  # 40 samples, 5 features
        y = np.array([0] * 20 + [1] * 20)  # 20 female, 20 male

        results = trainer.train(X, y)

        assert results is not None
        assert isinstance(results, TrainingResults)
        assert 0 <= results.accuracy <= 1
        assert 0 <= results.precision <= 1
        assert 0 <= results.recall <= 1
        assert hasattr(results, "cv_fold_scores")

    def test_train_imbalanced_dataset(self):
        """Should handle imbalanced datasets correctly."""
        trainer = SexClassifierTrainer()

        # Create imbalanced training data (70/30)
        X = np.random.randn(30, 5)
        y = np.array([1] * 21 + [0] * 9)  # 70% male, 30% female

        results = trainer.train(X, y)

        assert results is not None
        assert isinstance(results, TrainingResults)

    def test_train_with_custom_parameters(self):
        """Should train with custom hyperparameters."""
        config = ModelConfig(alpha=0.2, l1_ratio=0.8, cv_folds=5)
        trainer = SexClassifierTrainer(config=config)

        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)

        results = trainer.train(X, y)

        assert results is not None
        assert len(results.cv_fold_scores) == 5  # Should have 5 fold scores

    def test_train_stores_trained_model(self):
        """Should store trained model after training."""
        trainer = SexClassifierTrainer()

        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)

        trainer.train(X, y)

        assert trainer.trainer_model is not None
        assert hasattr(trainer.trainer_model, "predict_proba")

    def test_train_from_training_dataset(self):
        """Should train from TrainingDataset object."""
        trainer = SexClassifierTrainer()

        # Create TrainingDataset
        samples = [f"GSM{i:03d}" for i in range(40)]
        labels = np.array([0] * 20 + [1] * 20)
        dataset = TrainingDataset(samples=samples, labels=labels)

        # Create random expression data
        X = np.random.randn(40, 5)

        results = trainer.train(X, dataset.labels)

        assert results is not None
        assert results.accuracy >= 0  # Should have some accuracy


class TestSexClassifierTrainerCrossValidation:
    """Test cross-validation functionality."""

    def test_cv_folds_affect_results(self):
        """Should use correct number of CV folds."""
        config_5 = ModelConfig(cv_folds=5, random_state=42)
        config_10 = ModelConfig(cv_folds=10, random_state=42)

        trainer_5 = SexClassifierTrainer(config=config_5)
        trainer_10 = SexClassifierTrainer(config=config_10)

        X = np.random.randn(50, 5)
        y = np.array([0] * 25 + [1] * 25)

        results_5 = trainer_5.train(X, y)
        results_10 = trainer_10.train(X, y)

        assert len(results_5.cv_fold_scores) == 5
        assert len(results_10.cv_fold_scores) == 10

    def test_cv_stratification_maintains_balance(self):
        """Should use stratified k-fold to maintain class balance."""
        trainer = SexClassifierTrainer(ModelConfig(cv_folds=5))

        # Imbalanced data
        X = np.random.randn(30, 5)
        y = np.array([1] * 21 + [0] * 9)  # 70/30 split

        results = trainer.train(X, y)

        # Should handle imbalanced data with stratification
        assert results is not None
        assert all(0 <= score <= 1 for score in results.cv_fold_scores)


class TestSexClassifierTrainerPrediction:
    """Test prediction on trained model."""

    def test_predict_on_training_data(self):
        """Should make predictions on training data."""
        trainer = SexClassifierTrainer()

        X_train = np.random.randn(40, 5)
        y_train = np.array([0] * 20 + [1] * 20)

        trainer.train(X_train, y_train)

        # Make predictions
        predictions = trainer.predict_proba(X_train)

        assert predictions is not None
        assert predictions.shape[0] == 40  # 40 samples
        assert predictions.shape[1] == 2  # 2 classes

    def test_predict_proba_range(self):
        """Should return probabilities in [0, 1]."""
        trainer = SexClassifierTrainer()

        X_train = np.random.randn(40, 5)
        y_train = np.array([0] * 20 + [1] * 20)

        trainer.train(X_train, y_train)

        X_test = np.random.randn(10, 5)
        predictions = trainer.predict_proba(X_test)

        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_predict_without_training_raises_error(self):
        """Should raise error if predicting without training."""
        trainer = SexClassifierTrainer()

        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="not trained"):
            trainer.predict_proba(X)


class TestSexClassifierTrainerCheckpointing:
    """Test model checkpointing and recovery."""

    def test_save_checkpoint_creates_file(self):
        """Should save checkpoint to file."""
        trainer = SexClassifierTrainer()

        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)

        trainer.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model_checkpoint.pkl"

            trainer.save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

    def test_load_checkpoint_restores_model(self):
        """Should load checkpoint and restore model."""
        trainer1 = SexClassifierTrainer()

        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)

        trainer1.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model_checkpoint.pkl"

            # Save
            trainer1.save_checkpoint(str(checkpoint_path))

            # Load in new trainer
            trainer2 = SexClassifierTrainer()
            trainer2.load_checkpoint(str(checkpoint_path))

            # Predictions should match
            pred1 = trainer1.predict_proba(X)
            pred2 = trainer2.predict_proba(X)

            assert np.allclose(pred1, pred2)

    def test_checkpoint_preserves_config(self):
        """Should preserve configuration in checkpoint."""
        config = ModelConfig(alpha=0.3, l1_ratio=0.7)
        trainer1 = SexClassifierTrainer(config=config)

        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)

        trainer1.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model_checkpoint.pkl"

            trainer1.save_checkpoint(str(checkpoint_path))

            trainer2 = SexClassifierTrainer()
            trainer2.load_checkpoint(str(checkpoint_path))

            # Config should be preserved
            assert trainer2.config.alpha == 0.3
            assert trainer2.config.l1_ratio == 0.7


class TestSexClassifierTrainerIntegration:
    """Integration tests for full training workflow."""

    def test_train_validate_checkpoint_pipeline(self):
        """Should support complete train-validate-save pipeline."""
        config = ModelConfig(alpha=0.5, l1_ratio=0.5, cv_folds=5)
        trainer = SexClassifierTrainer(config=config)

        # Generate train/test split
        X_train = np.random.randn(40, 5)
        y_train = np.array([0] * 20 + [1] * 20)

        # Train
        results = trainer.train(X_train, y_train)

        assert results is not None
        assert results.accuracy >= 0

        # Make predictions
        X_test = np.random.randn(10, 5)
        predictions = trainer.predict_proba(X_test)

        assert predictions.shape[0] == 10

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "final_model.pkl"
            trainer.save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

            # Load and verify
            trainer2 = SexClassifierTrainer()
            trainer2.load_checkpoint(str(checkpoint_path))

            pred2 = trainer2.predict_proba(X_test)
            assert np.allclose(predictions, pred2)
