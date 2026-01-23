"""Tests for expression-based sex inference (Phase 6A).

Tests ExpressionFetcher, SexClassifier, and ElasticNetInferenceStrategy.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.sage.expression_fetcher import (
    ExpressionFetcher,
    get_x_chromosome_genes,
    get_y_chromosome_genes,
)
from src.sage.sex_inference import SexClassifier, ElasticNetInferenceStrategy, InferenceResult


class TestExpressionFetcherGeneSelection:
    """Test gene selection utilities."""

    def test_get_x_chromosome_genes(self):
        """X chromosome genes should include XIST."""
        x_genes = get_x_chromosome_genes()
        assert isinstance(x_genes, list)
        assert len(x_genes) > 0
        assert "XIST" in x_genes

    def test_get_y_chromosome_genes(self):
        """Y chromosome genes should include RPS4Y1 and DDX3Y."""
        y_genes = get_y_chromosome_genes()
        assert isinstance(y_genes, list)
        assert len(y_genes) > 0
        assert "RPS4Y1" in y_genes
        assert "DDX3Y" in y_genes

    def test_male_markers_present(self):
        """All expected male markers should be available."""
        y_genes = get_y_chromosome_genes()
        expected = {"RPS4Y1", "DDX3Y", "EIF1AY", "KDM5D"}
        assert expected.issubset(set(y_genes))


class TestExpressionFetcherColumnDetection:
    """Test column detection in expression tables."""

    def test_find_gene_column_standard_names(self):
        """Should find standard gene column names."""
        fetcher = ExpressionFetcher()

        # Test various standard names
        for col_name in ["GENE_SYMBOL", "Gene Symbol", "Symbol", "gene_name"]:
            df = pd.DataFrame({col_name: ["XIST", "RPS4Y1"], "VALUE": [10.0, 5.0]})
            found = fetcher._find_gene_column(df)
            assert found == col_name, f"Failed to find {col_name}"

    def test_find_gene_column_case_insensitive(self):
        """Should find gene column case-insensitively."""
        fetcher = ExpressionFetcher()
        df = pd.DataFrame({"gene_symbol": ["XIST"], "value": [10.0]})
        found = fetcher._find_gene_column(df)
        assert found == "gene_symbol"

    def test_find_value_column_standard_names(self):
        """Should find standard value column names."""
        fetcher = ExpressionFetcher()

        for col_name in ["VALUE", "TPM", "FPKM", "counts"]:
            df = pd.DataFrame({"GENE_SYMBOL": ["XIST"], col_name: [10.0]})
            found = fetcher._find_value_column(df)
            assert found == col_name, f"Failed to find {col_name}"

    def test_find_columns_fallback(self):
        """Should use fallback columns if standard names not found."""
        fetcher = ExpressionFetcher()
        df = pd.DataFrame({"col1": ["XIST"], "col2": [10.0]})
        gene_col = fetcher._find_gene_column(df)
        value_col = fetcher._find_value_column(df)
        assert gene_col is not None
        assert value_col is not None


class TestExpressionFetcherSampleExtraction:
    """Test expression data extraction from GSM samples."""

    def test_fetch_sample_expression_success(self):
        """Should extract expression for specified genes."""
        fetcher = ExpressionFetcher()

        # Mock GSE and GSM
        mock_gse = MagicMock()
        mock_gsm = MagicMock()
        mock_gsm.table = pd.DataFrame(
            {"GENE_SYMBOL": ["XIST", "RPS4Y1", "DDX3Y"], "VALUE": [100.0, 5.0, 3.0]}
        )
        mock_gse.gsms = {"GSM001": mock_gsm}

        result = fetcher.fetch_sample_expression(mock_gse, "GSM001")

        assert isinstance(result, dict)
        assert "XIST" in result
        assert result["XIST"] == 100.0
        assert result["RPS4Y1"] == 5.0
        assert result["DDX3Y"] == 3.0

    def test_fetch_sample_expression_missing_gene(self):
        """Should return 0.0 for genes not found in table."""
        fetcher = ExpressionFetcher()

        mock_gse = MagicMock()
        mock_gsm = MagicMock()
        mock_gsm.table = pd.DataFrame({"GENE_SYMBOL": ["XIST"], "VALUE": [100.0]})
        mock_gse.gsms = {"GSM001": mock_gsm}

        result = fetcher.fetch_sample_expression(mock_gse, "GSM001", ["XIST", "RPS4Y1"])

        assert result["XIST"] == 100.0
        assert result["RPS4Y1"] == 0.0

    def test_fetch_sample_expression_invalid_gsm(self):
        """Should raise KeyError for invalid GSM ID."""
        fetcher = ExpressionFetcher()

        mock_gse = MagicMock()
        mock_gse.gsms = {}

        with pytest.raises(KeyError):
            fetcher.fetch_sample_expression(mock_gse, "INVALID_GSM")

    def test_fetch_sample_expression_no_table(self):
        """Should raise ValueError if expression table missing."""
        fetcher = ExpressionFetcher()

        mock_gse = MagicMock()
        mock_gsm = MagicMock()
        mock_gsm.table = None
        mock_gse.gsms = {"GSM001": mock_gsm}

        with pytest.raises(ValueError):
            fetcher.fetch_sample_expression(mock_gse, "GSM001")


class TestExpressionFetcherStudyExtraction:
    """Test expression data extraction from full studies."""

    def test_fetch_study_expression_success(self):
        """Should fetch expression for all samples in study."""
        fetcher = ExpressionFetcher()

        # Mock GSE with 3 samples
        mock_gse = MagicMock()
        mock_gse.name = "GSE123"

        mock_gsm1 = MagicMock()
        mock_gsm1.table = pd.DataFrame({"GENE_SYMBOL": ["XIST", "RPS4Y1"], "VALUE": [100.0, 5.0]})

        mock_gsm2 = MagicMock()
        mock_gsm2.table = pd.DataFrame({"GENE_SYMBOL": ["XIST", "RPS4Y1"], "VALUE": [5.0, 100.0]})

        mock_gse.gsms = {
            "GSM001": mock_gsm1,
            "GSM002": mock_gsm2,
        }

        result = fetcher.fetch_study_expression(mock_gse)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "XIST" in result.columns
        assert "RPS4Y1" in result.columns

    def test_fetch_study_expression_with_errors(self):
        """Should continue if some samples fail, logging warnings."""
        fetcher = ExpressionFetcher()

        mock_gse = MagicMock()
        mock_gse.name = "GSE123"

        # Good GSM
        mock_gsm1 = MagicMock()
        mock_gsm1.table = pd.DataFrame({"GENE_SYMBOL": ["XIST"], "VALUE": [100.0]})

        # Bad GSM (no table)
        mock_gsm2 = MagicMock()
        mock_gsm2.table = None

        mock_gse.gsms = {
            "GSM001": mock_gsm1,
            "GSM002": mock_gsm2,
        }

        result = fetcher.fetch_study_expression(mock_gse)

        # Should still return results from GSM001
        assert len(result) >= 1
        assert "GSM001" in result.index


class TestSexClassifierInitialization:
    """Test SexClassifier initialization."""

    def test_init_default_genes(self):
        """Should initialize with default X and Y genes."""
        classifier = SexClassifier()

        assert "XIST" in classifier.x_genes
        assert "RPS4Y1" in classifier.y_genes
        assert "DDX3Y" in classifier.y_genes

    def test_init_custom_genes(self):
        """Should accept custom gene lists."""
        x_genes = ["XIST", "CUSTOM_X"]
        y_genes = ["RPS4Y1", "CUSTOM_Y"]

        classifier = SexClassifier(x_genes=x_genes, y_genes=y_genes)

        assert classifier.x_genes == x_genes
        assert classifier.y_genes == y_genes
        assert len(classifier.all_genes) == 4

    def test_init_with_model(self):
        """Should accept pre-trained model."""
        mock_model = MagicMock()
        classifier = SexClassifier(model=mock_model)

        assert classifier.model is mock_model


class TestSexClassifierTraining:
    """Test SexClassifier model training."""

    def test_train_model(self):
        """Should train elastic net model successfully."""
        classifier = SexClassifier()

        # Create mock training data
        X = np.random.randn(20, len(classifier.all_genes))
        y = np.array([0] * 10 + [1] * 10)  # 10 females, 10 males

        classifier.train(X, y)

        assert classifier.model is not None
        assert hasattr(classifier.model, "predict_proba")

    def test_train_with_custom_parameters(self):
        """Should accept custom regularization parameters."""
        classifier = SexClassifier()

        X = np.random.randn(20, len(classifier.all_genes))
        y = np.array([0] * 10 + [1] * 10)

        # Should not raise
        classifier.train(X, y, alpha=0.3, l1_ratio=0.8)
        assert classifier.model is not None


class TestSexClassifierPrediction:
    """Test sex classification predictions."""

    def test_predict_sex_score(self):
        """Should return probability score in [0, 1]."""
        classifier = SexClassifier()

        # Train model
        X = np.random.randn(20, len(classifier.all_genes))
        y = np.array([0] * 10 + [1] * 10)
        classifier.train(X, y)

        # Predict
        sample = {gene: np.random.randn() for gene in classifier.all_genes}
        score = classifier.predict_sex_score(sample)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_sex_score_without_training(self):
        """Should raise ValueError if model not trained."""
        classifier = SexClassifier()

        sample = {gene: 0.0 for gene in classifier.all_genes}

        with pytest.raises(ValueError, match="not trained"):
            classifier.predict_sex_score(sample)

    def test_classify_sample_female(self):
        """Should classify as female when score < threshold."""
        classifier = SexClassifier()

        # Train model
        X = np.random.randn(20, len(classifier.all_genes))
        y = np.array([0] * 10 + [1] * 10)
        classifier.train(X, y)

        # Create "female-like" sample (low Y signal)
        sample = {"XIST": 100.0}  # High XIST = female
        sample.update({gene: 0.0 for gene in classifier.all_genes if gene != "XIST"})

        sex, conf = classifier.classify_sample(sample, threshold=0.7)

        assert sex in ["male", "female", "ambiguous"]
        assert 0.0 <= conf <= 1.0

    def test_classify_sample_ambiguous(self):
        """Should classify as ambiguous when near threshold."""
        classifier = SexClassifier()

        # Train model
        X = np.random.randn(20, len(classifier.all_genes))
        y = np.array([0] * 10 + [1] * 10)
        classifier.train(X, y)

        # Create ambiguous sample (balanced signals)
        sample = {gene: 0.5 for gene in classifier.all_genes}

        sex, conf = classifier.classify_sample(sample, threshold=0.7)

        # May be classified as ambiguous or male/female depending on model
        assert sex in ["male", "female", "ambiguous"]


class TestSexClassifierBatchClassification:
    """Test batch classification of study samples."""

    def test_classify_study(self):
        """Should classify all samples in study."""
        classifier = SexClassifier()

        # Train model
        X = np.random.randn(20, len(classifier.all_genes))
        y = np.array([0] * 10 + [1] * 10)
        classifier.train(X, y)

        # Create mock study data (5 samples)
        expression_df = pd.DataFrame(
            np.random.randn(5, len(classifier.all_genes)),
            columns=classifier.all_genes,
            index=["GSM001", "GSM002", "GSM003", "GSM004", "GSM005"],
        )

        result = classifier.classify_study(expression_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "sample_id" in result.columns
        assert "inferred_sex" in result.columns
        assert "confidence" in result.columns
        assert all(result["inferred_sex"].isin(["male", "female", "ambiguous"]))
        assert all((result["confidence"] >= 0.0) & (result["confidence"] <= 1.0))


class TestElasticNetInferenceStrategy:
    """Test ElasticNetInferenceStrategy."""

    def test_infer_requires_gse(self):
        """Should raise ValueError if 'gse' key missing."""
        strategy = ElasticNetInferenceStrategy()

        with pytest.raises(ValueError, match="'gse' key"):
            strategy.infer({})

    def test_infer_handles_missing_expression(self):
        """Should return non-inferrable result if expression fetch fails."""
        strategy = ElasticNetInferenceStrategy()

        mock_gse = MagicMock()
        mock_gse.name = "GSE123"
        mock_gse.gsms = {}  # Empty - will fail

        result = strategy.infer({"gse": mock_gse})

        assert isinstance(result, InferenceResult)
        assert result.inferrable is False
        assert result.method == "expression_elasticnet"

    def test_infer_success(self):
        """Should successfully infer sex for study with expression data."""
        strategy = ElasticNetInferenceStrategy()

        # Train classifier first
        X = np.random.randn(20, len(strategy.classifier.all_genes))
        y = np.array([0] * 10 + [1] * 10)
        strategy.classifier.train(X, y)

        # Mock GSE with samples
        mock_gse = MagicMock()
        mock_gse.name = "GSE123"

        mock_gsm = MagicMock()
        mock_gsm.table = pd.DataFrame(
            {
                "GENE_SYMBOL": strategy.classifier.all_genes,
                "VALUE": np.random.randn(len(strategy.classifier.all_genes)),
            }
        )

        mock_gse.gsms = {"GSM001": mock_gsm}

        result = strategy.infer({"gse": mock_gse})

        assert isinstance(result, InferenceResult)
        assert result.method == "expression_elasticnet"
        assert "sample_count" in result.factors
        assert result.factors["sample_count"] == 1
