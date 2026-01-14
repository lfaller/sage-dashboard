"""Tests for sex inference module."""
from sage.sex_inference import (
    analyze_sample_names,
    calculate_confidence,
    infer_from_metadata,
    InferenceStrategy,
    MetadataInferenceStrategy,
    InferenceResult,
)


class TestAnalyzeSampleNames:
    """Tests for sample name analysis and pattern detection."""

    def test_clear_male_female_patterns(self):
        """Test detection of clear M/F patterns."""
        sample_names = ["M1", "M2", "F1", "F2"]
        result = analyze_sample_names(sample_names)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["male_count"] == 2
        assert result["female_count"] == 2

    def test_full_male_female_words(self):
        """Test detection of full 'Male'/'Female' words."""
        sample_names = ["Male_001", "Male_002", "Female_001", "Female_002"]
        result = analyze_sample_names(sample_names)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["male_count"] == 2
        assert result["female_count"] == 2

    def test_case_insensitive_matching(self):
        """Test case-insensitive pattern matching."""
        sample_names = ["male", "MALE", "female", "FEMALE"]
        result = analyze_sample_names(sample_names)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["male_count"] == 2
        assert result["female_count"] == 2

    def test_mixed_delimiters(self):
        """Test detection with various delimiters."""
        sample_names = ["M-1", "M_2", "M.3", "F-1", "F_2", "F.3"]
        result = analyze_sample_names(sample_names)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["male_count"] == 3
        assert result["female_count"] == 3

    def test_ambiguous_samples(self):
        """Test samples with no clear sex pattern."""
        sample_names = ["Sample1", "Sample2", "Sample3"]
        result = analyze_sample_names(sample_names)

        assert result["pattern"] == "none"
        assert result["confidence"] == 0.0
        assert result["male_count"] == 0
        assert result["female_count"] == 0

    def test_partial_labeling(self):
        """Test partial labeling (50% have sex labels)."""
        sample_names = ["M1", "F1", "Sample3", "Sample4"]
        result = analyze_sample_names(sample_names)

        assert result["pattern"] == "partial"
        assert result["confidence"] == 0.5
        assert result["male_count"] == 1
        assert result["female_count"] == 1

    def test_empty_sample_list(self):
        """Test handling of empty sample list."""
        result = analyze_sample_names([])

        assert result["pattern"] == "none"
        assert result["confidence"] == 0.0
        assert result["male_count"] == 0
        assert result["female_count"] == 0

    def test_none_values_in_list(self):
        """Test handling of None values in sample list."""
        sample_names = [None, "M1", "F1"]
        result = analyze_sample_names(sample_names)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["male_count"] == 1
        assert result["female_count"] == 1


class TestCalculateConfidence:
    """Tests for confidence score calculation."""

    def test_high_confidence_all_factors(self):
        """Test high confidence when all positive factors present."""
        factors = {
            "is_rna_seq": True,
            "has_sufficient_samples": True,
            "is_human": True,
            "sample_name_confidence": 1.0,
        }
        result = calculate_confidence(factors)

        assert 0.9 <= result <= 1.0
        assert result > 0.8

    def test_zero_confidence_no_factors(self):
        """Test zero confidence when no positive factors present."""
        factors = {
            "is_rna_seq": False,
            "has_sufficient_samples": False,
            "is_human": False,
            "sample_name_confidence": 0.0,
        }
        result = calculate_confidence(factors)

        assert result == 0.0

    def test_rna_seq_bonus(self):
        """Test RNA-seq study gets confidence boost."""
        factors_with_rna = {
            "is_rna_seq": True,
            "has_sufficient_samples": False,
            "is_human": False,
            "sample_name_confidence": 0.0,
        }
        factors_without_rna = {
            "is_rna_seq": False,
            "has_sufficient_samples": False,
            "is_human": False,
            "sample_name_confidence": 0.0,
        }

        result_with = calculate_confidence(factors_with_rna)
        result_without = calculate_confidence(factors_without_rna)

        assert result_with > result_without
        assert result_with == 0.35

    def test_sample_size_threshold(self):
        """Test sample size threshold effect."""
        factors_sufficient = {
            "is_rna_seq": False,
            "has_sufficient_samples": True,
            "is_human": False,
            "sample_name_confidence": 0.0,
        }
        factors_insufficient = {
            "is_rna_seq": False,
            "has_sufficient_samples": False,
            "is_human": False,
            "sample_name_confidence": 0.0,
        }

        result_sufficient = calculate_confidence(factors_sufficient)
        result_insufficient = calculate_confidence(factors_insufficient)

        assert result_sufficient > result_insufficient
        assert result_sufficient == 0.25

    def test_human_organism_bonus(self):
        """Test human organism gets bonus."""
        factors_human = {
            "is_rna_seq": False,
            "has_sufficient_samples": False,
            "is_human": True,
            "sample_name_confidence": 0.0,
        }
        factors_non_human = {
            "is_rna_seq": False,
            "has_sufficient_samples": False,
            "is_human": False,
            "sample_name_confidence": 0.0,
        }

        result_human = calculate_confidence(factors_human)
        result_non_human = calculate_confidence(factors_non_human)

        assert result_human > result_non_human
        assert result_human == 0.15

    def test_partial_sample_names(self):
        """Test partial sample name confidence contribution."""
        factors = {
            "is_rna_seq": False,
            "has_sufficient_samples": False,
            "is_human": False,
            "sample_name_confidence": 0.5,
        }
        result = calculate_confidence(factors)

        # 0.5 * 0.25 = 0.125
        assert result == 0.125

    def test_boundary_values(self):
        """Test boundary values (0.0 and 1.0)."""
        factors_min = {
            "is_rna_seq": False,
            "has_sufficient_samples": False,
            "is_human": False,
            "sample_name_confidence": 0.0,
        }
        factors_max = {
            "is_rna_seq": True,
            "has_sufficient_samples": True,
            "is_human": True,
            "sample_name_confidence": 1.0,
        }

        result_min = calculate_confidence(factors_min)
        result_max = calculate_confidence(factors_max)

        assert result_min == 0.0
        assert 0.0 <= result_min <= 1.0
        assert 0.0 <= result_max <= 1.0


class TestInferFromMetadata:
    """Tests for metadata-based inference."""

    def test_ideal_rna_seq_study(self):
        """Test ideal rescue candidate (RNA-seq, large sample, named samples)."""
        study_dict = {
            "study_type": "RNA-seq",
            "sample_count": 50,
            "organism": "Homo sapiens",
            "has_sex_metadata": False,
            "sample_names": ["M1", "M2", "F1", "F2"],
        }
        result = infer_from_metadata(study_dict)

        assert result["sex_inferrable"] is True
        assert result["sex_inference_confidence"] >= 0.5
        assert result["inference_method"] == "metadata"
        assert "inference_factors" in result

    def test_microarray_study(self):
        """Test microarray study (lower confidence expected)."""
        study_dict = {
            "study_type": "microarray",
            "sample_count": 30,
            "organism": "Homo sapiens",
            "has_sex_metadata": False,
            "sample_names": [],
        }
        result = infer_from_metadata(study_dict)

        assert "sex_inferrable" in result
        assert "sex_inference_confidence" in result
        assert 0.0 <= result["sex_inference_confidence"] <= 1.0

    def test_small_sample_count(self):
        """Test study with <20 samples."""
        study_dict = {
            "study_type": "RNA-seq",
            "sample_count": 10,
            "organism": "Homo sapiens",
            "has_sex_metadata": False,
            "sample_names": [],  # No sample names to reduce confidence
        }
        result = infer_from_metadata(study_dict)

        # Should have lower confidence due to small sample size
        assert result["sex_inference_confidence"] < 0.7

    def test_no_sample_names(self):
        """Test study with missing sample_names field."""
        study_dict = {
            "study_type": "RNA-seq",
            "sample_count": 50,
            "organism": "Homo sapiens",
            "has_sex_metadata": False,
        }
        result = infer_from_metadata(study_dict)

        assert "sex_inferrable" in result
        assert "sex_inference_confidence" in result

    def test_non_human_organism(self):
        """Test non-human organism (mouse, rat, etc)."""
        study_dict = {
            "study_type": "RNA-seq",
            "sample_count": 50,
            "organism": "Mus musculus",
            "has_sex_metadata": False,
            "sample_names": ["M1", "M2", "F1", "F2"],
        }
        result = infer_from_metadata(study_dict)

        assert "sex_inferrable" in result
        # Non-human should have lower confidence than human
        assert result["sex_inference_confidence"] < 0.95

    def test_missing_study_type(self):
        """Test study with missing study_type field."""
        study_dict = {
            "sample_count": 50,
            "organism": "Homo sapiens",
            "has_sex_metadata": False,
            "sample_names": ["M1", "F1"],
        }
        result = infer_from_metadata(study_dict)

        assert "sex_inferrable" in result
        assert "sex_inference_confidence" in result

    def test_returns_required_fields(self):
        """Test that result has all required fields."""
        study_dict = {
            "study_type": "RNA-seq",
            "sample_count": 30,
            "organism": "Homo sapiens",
            "has_sex_metadata": False,
        }
        result = infer_from_metadata(study_dict)

        assert "sex_inferrable" in result
        assert "sex_inference_confidence" in result
        assert "inference_method" in result
        assert "inference_factors" in result

    def test_confidence_range(self):
        """Test confidence value is in valid range [0.0, 1.0]."""
        study_dict = {
            "study_type": "RNA-seq",
            "sample_count": 30,
            "organism": "Homo sapiens",
            "has_sex_metadata": False,
            "sample_names": ["M1", "F1"],
        }
        result = infer_from_metadata(study_dict)

        assert 0.0 <= result["sex_inference_confidence"] <= 1.0


class TestInferenceStrategy:
    """Tests for inference strategy pattern."""

    def test_strategy_interface_exists(self):
        """Test that InferenceStrategy abstract base class exists."""
        assert hasattr(InferenceStrategy, "infer")
        assert InferenceStrategy.__abstractmethods__

    def test_metadata_strategy_implements_interface(self):
        """Test that MetadataInferenceStrategy properly implements interface."""
        strategy = MetadataInferenceStrategy()
        assert isinstance(strategy, InferenceStrategy)
        assert hasattr(strategy, "infer")

    def test_can_add_new_strategy(self):
        """Test that new strategies can be added by subclassing."""

        class CustomStrategy(InferenceStrategy):
            def infer(self, study_dict):
                return InferenceResult(inferrable=True, confidence=0.5, method="custom", factors={})

        strategy = CustomStrategy()
        result = strategy.infer({})

        assert isinstance(result, InferenceResult)
        assert result.method == "custom"
