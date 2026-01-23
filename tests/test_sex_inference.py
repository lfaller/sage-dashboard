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


# ============================================================================
# Tests for Characteristics-Based Sex Detection
# ============================================================================


class TestExtractSexFromCharacteristics:
    """Tests for extracting sex from characteristics_ch1."""

    def test_standard_colon_format_male(self):
        """Test standard 'key: value' format with male."""
        from sage.sex_inference import extract_sex_from_characteristics

        result = extract_sex_from_characteristics(["sex: male"])
        assert result == "male"

    def test_standard_colon_format_female(self):
        """Test standard 'key: value' format with female."""
        from sage.sex_inference import extract_sex_from_characteristics

        result = extract_sex_from_characteristics(["sex: female"])
        assert result == "female"

    def test_case_insensitive_key(self):
        """Test case-insensitive key matching."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["SEX: male"]) == "male"
        assert extract_sex_from_characteristics(["Sex: Female"]) == "female"

    def test_case_insensitive_value(self):
        """Test case-insensitive value matching."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["sex: MALE"]) == "male"
        assert extract_sex_from_characteristics(["sex: Female"]) == "female"

    def test_gender_key(self):
        """Test 'gender' as alternative key."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["gender: male"]) == "male"
        assert extract_sex_from_characteristics(["gender: female"]) == "female"

    def test_abbreviated_values_m_f(self):
        """Test M/F abbreviations."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["sex: M"]) == "male"
        assert extract_sex_from_characteristics(["sex: F"]) == "female"
        assert extract_sex_from_characteristics(["gender: m"]) == "male"
        assert extract_sex_from_characteristics(["gender: f"]) == "female"

    def test_alternate_delimiter_equals(self):
        """Test equals sign as delimiter."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["sex=male"]) == "male"
        assert extract_sex_from_characteristics(["sex=female"]) == "female"

    def test_alternate_delimiter_pipe(self):
        """Test pipe as delimiter."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["sex | male"]) == "male"
        assert extract_sex_from_characteristics(["sex|female"]) == "female"

    def test_alternate_key_sample_sex(self):
        """Test 'sample_sex' as key."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["sample_sex: male"]) == "male"
        assert extract_sex_from_characteristics(["sample_sex: female"]) == "female"

    def test_alternate_key_sex_ch1(self):
        """Test 'sex_ch1' as key."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["sex_ch1: male"]) == "male"
        assert extract_sex_from_characteristics(["sex_ch1: female"]) == "female"

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics(["sex : male"]) == "male"
        assert extract_sex_from_characteristics(["sex:  female  "]) == "female"
        assert extract_sex_from_characteristics(["  sex: male  "]) == "male"

    def test_mixed_characteristics(self):
        """Test extraction from list with non-sex fields."""
        from sage.sex_inference import extract_sex_from_characteristics

        chars = ["tissue: brain", "sex: male", "age: 45"]
        assert extract_sex_from_characteristics(chars) == "male"

    def test_no_sex_field(self):
        """Test when no sex field present."""
        from sage.sex_inference import extract_sex_from_characteristics

        chars = ["tissue: liver", "age: 30"]
        assert extract_sex_from_characteristics(chars) is None

    def test_empty_characteristics(self):
        """Test with empty list."""
        from sage.sex_inference import extract_sex_from_characteristics

        assert extract_sex_from_characteristics([]) is None

    def test_conflicting_values(self):
        """Test when multiple conflicting sex fields exist."""
        from sage.sex_inference import extract_sex_from_characteristics

        chars = ["sex: male", "gender: female"]
        assert extract_sex_from_characteristics(chars) is None

    def test_duplicate_consistent_values(self):
        """Test when multiple sex fields agree."""
        from sage.sex_inference import extract_sex_from_characteristics

        chars = ["sex: male", "gender: male"]
        # Should return the value (consistent)
        assert extract_sex_from_characteristics(chars) == "male"

    def test_malformed_characteristics(self):
        """Test handling of malformed characteristics."""
        from sage.sex_inference import extract_sex_from_characteristics

        # No delimiter
        assert extract_sex_from_characteristics(["sexmale"]) is None
        # Empty value
        assert extract_sex_from_characteristics(["sex: "]) is None


class TestAnalyzeSampleCharacteristics:
    """Tests for analyzing characteristics across samples."""

    def test_clear_pattern_all_labeled(self):
        """Test clear pattern when all samples have sex metadata."""
        from sage.sex_inference import analyze_sample_characteristics

        samples = [["sex: male", "age: 30"], ["sex: female", "age: 25"]]
        result = analyze_sample_characteristics(samples)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["male_count"] == 1
        assert result["female_count"] == 1
        assert result["source"] == "characteristics"

    def test_partial_pattern_half_labeled(self):
        """Test partial pattern with 50% labeling."""
        from sage.sex_inference import analyze_sample_characteristics

        samples = [["sex: male", "age: 30"], ["tissue: brain"]]  # No sex field
        result = analyze_sample_characteristics(samples)

        assert result["pattern"] == "partial"
        assert result["confidence"] == 0.5
        assert result["male_count"] == 1
        assert result["female_count"] == 0

    def test_partial_pattern_above_threshold(self):
        """Test partial pattern above 50% threshold."""
        from sage.sex_inference import analyze_sample_characteristics

        samples = [["sex: male"], ["sex: female"], ["sex: male"], ["tissue: brain"]]  # No sex field
        result = analyze_sample_characteristics(samples)

        assert result["pattern"] == "partial"
        assert result["confidence"] == 0.75
        assert result["male_count"] == 2
        assert result["female_count"] == 1

    def test_none_pattern_no_metadata(self):
        """Test none pattern when no samples have sex."""
        from sage.sex_inference import analyze_sample_characteristics

        samples = [["tissue: brain"], ["age: 45"]]
        result = analyze_sample_characteristics(samples)

        assert result["pattern"] == "none"
        assert result["confidence"] == 0.0
        assert result["male_count"] == 0
        assert result["female_count"] == 0

    def test_mixed_formats(self):
        """Test various characteristic formats work together."""
        from sage.sex_inference import analyze_sample_characteristics

        samples = [["sex: male"], ["gender: F"], ["sample_sex: Male"]]
        result = analyze_sample_characteristics(samples)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["male_count"] == 2
        assert result["female_count"] == 1

    def test_empty_samples_list(self):
        """Test with empty samples list."""
        from sage.sex_inference import analyze_sample_characteristics

        result = analyze_sample_characteristics([])

        assert result["pattern"] == "none"
        assert result["confidence"] == 0.0

    def test_all_empty_characteristics(self):
        """Test when all samples have empty characteristics."""
        from sage.sex_inference import analyze_sample_characteristics

        samples = [[], [], []]
        result = analyze_sample_characteristics(samples)

        assert result["pattern"] == "none"
        assert result["confidence"] == 0.0

    def test_conflicting_characteristics_skipped(self):
        """Test that samples with conflicting sex are not counted."""
        from sage.sex_inference import analyze_sample_characteristics

        samples = [
            ["sex: male", "gender: female"],  # Conflicting - not counted
            ["sex: male"],
            ["sex: female"],
        ]
        result = analyze_sample_characteristics(samples)

        # First sample conflict is skipped, so only 2 out of 3 are labeled
        assert result["male_count"] == 1
        assert result["female_count"] == 1
        assert result["confidence"] == 2 / 3


class TestMergeSexAnalyses:
    """Tests for merging characteristics and sample name analyses."""

    def test_characteristics_wins_when_clear(self):
        """Test characteristics takes priority when clear."""
        from sage.sex_inference import merge_sex_analyses

        chars = {
            "pattern": "clear",
            "confidence": 1.0,
            "male_count": 3,
            "female_count": 3,
            "source": "characteristics",
        }
        names = {"pattern": "partial", "confidence": 0.5, "male_count": 2, "female_count": 1}

        result = merge_sex_analyses(chars, names)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["source"] == "characteristics"

    def test_sample_names_fallback_when_chars_empty(self):
        """Test sample names used when characteristics empty."""
        from sage.sex_inference import merge_sex_analyses

        chars = {
            "pattern": "none",
            "confidence": 0.0,
            "male_count": 0,
            "female_count": 0,
            "source": "characteristics",
        }
        names = {"pattern": "clear", "confidence": 1.0, "male_count": 4, "female_count": 4}

        result = merge_sex_analyses(chars, names)

        assert result["pattern"] == "clear"
        assert result["confidence"] == 1.0
        assert result["source"] == "sample_names"

    def test_both_sources_empty(self):
        """Test when both sources have no data."""
        from sage.sex_inference import merge_sex_analyses

        chars = {
            "pattern": "none",
            "confidence": 0.0,
            "male_count": 0,
            "female_count": 0,
            "source": "characteristics",
        }
        names = {"pattern": "none", "confidence": 0.0, "male_count": 0, "female_count": 0}

        result = merge_sex_analyses(chars, names)

        assert result["pattern"] == "none"
        assert result["confidence"] == 0.0

    def test_characteristics_higher_confidence(self):
        """Test characteristics prioritized even when not clear."""
        from sage.sex_inference import merge_sex_analyses

        chars = {
            "pattern": "partial",
            "confidence": 0.75,
            "male_count": 3,
            "female_count": 1,
            "source": "characteristics",
        }
        names = {"pattern": "partial", "confidence": 0.5, "male_count": 2, "female_count": 1}

        result = merge_sex_analyses(chars, names)

        # Characteristics should win since confidence >= 0.5
        assert result["source"] == "characteristics"
        assert result["male_count"] == 3
        assert result["female_count"] == 1

    def test_names_wins_when_characteristics_partial_and_names_higher(self):
        """Test names wins if characteristics is low and names is higher."""
        from sage.sex_inference import merge_sex_analyses

        chars = {
            "pattern": "partial",
            "confidence": 0.25,  # Below threshold
            "male_count": 1,
            "female_count": 0,
            "source": "characteristics",
        }
        names = {"pattern": "clear", "confidence": 1.0, "male_count": 10, "female_count": 10}

        result = merge_sex_analyses(chars, names)

        # Names should win since confidence >= 0.5 and characteristics < 0.5
        assert result["source"] == "sample_names"
        assert result["confidence"] == 1.0
