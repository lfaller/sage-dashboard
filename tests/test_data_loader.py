"""Tests for data loading module."""
import pytest
from unittest.mock import Mock, patch

from sage.data_loader import (
    Study,
    parse_geo_metadata,
    calculate_sex_metadata_completeness,
    upsert_studies,
)


class TestStudyDataClass:
    """Tests for Study dataclass."""

    def test_study_initialization(self):
        """Test Study can be initialized with required fields."""
        study = Study(
            geo_accession="GSE123456",
            title="Test Study",
            organism="Homo sapiens",
            sample_count=50,
        )
        assert study.geo_accession == "GSE123456"
        assert study.title == "Test Study"
        assert study.organism == "Homo sapiens"
        assert study.sample_count == 50
        assert study.has_sex_metadata is False
        assert study.sex_metadata_completeness == 0.0

    def test_study_with_optional_fields(self):
        """Test Study with optional fields."""
        study = Study(
            geo_accession="GSE123456",
            title="Test Study",
            organism="Homo sapiens",
            sample_count=50,
            summary="A test study",
            platform="GPL570",
            study_type="RNA-seq",
        )
        assert study.summary == "A test study"
        assert study.platform == "GPL570"
        assert study.study_type == "RNA-seq"


class TestSexMetadataCompleteness:
    """Tests for sex metadata completeness calculation."""

    def test_all_samples_have_sex(self):
        """Test when all samples have sex metadata."""
        completeness = calculate_sex_metadata_completeness(samples_with_sex=100, total_samples=100)
        assert completeness == 1.0

    def test_no_samples_have_sex(self):
        """Test when no samples have sex metadata."""
        completeness = calculate_sex_metadata_completeness(samples_with_sex=0, total_samples=100)
        assert completeness == 0.0

    def test_partial_sex_metadata(self):
        """Test when some samples have sex metadata."""
        completeness = calculate_sex_metadata_completeness(samples_with_sex=50, total_samples=100)
        assert completeness == 0.5

    def test_zero_samples(self):
        """Test with zero samples."""
        completeness = calculate_sex_metadata_completeness(samples_with_sex=0, total_samples=0)
        assert completeness == 0.0

    def test_more_with_sex_than_total(self):
        """Test validation - samples with sex can't exceed total."""
        with pytest.raises(ValueError):
            calculate_sex_metadata_completeness(samples_with_sex=150, total_samples=100)


class TestParseGeoMetadata:
    """Tests for GEO metadata parsing."""

    def test_parse_valid_geo_response(self):
        """Test parsing a valid GEO API response."""
        geo_data = {
            "accession": "GSE123456",
            "title": "Test Study",
            "summary": "A test study",
            "gse": {
                "organism": "Homo sapiens",
                "overall_design": "Test design",
                "type": ["Expression profiling by high throughput sequencing"],
                "sample_id": ["GSM1", "GSM2", "GSM3"],
            },
        }

        study = parse_geo_metadata(geo_data)

        assert study.geo_accession == "GSE123456"
        assert study.title == "Test Study"
        assert study.organism == "Homo sapiens"
        assert study.sample_count == 3

    def test_parse_geo_with_minimal_data(self):
        """Test parsing GEO data with minimal fields."""
        geo_data = {
            "accession": "GSE123456",
            "title": "Test Study",
            "gse": {
                "organism": "Homo sapiens",
                "sample_id": ["GSM1", "GSM2"],
            },
        }

        study = parse_geo_metadata(geo_data)

        assert study.geo_accession == "GSE123456"
        assert study.title == "Test Study"
        assert study.sample_count == 2

    def test_parse_geo_missing_required_field(self):
        """Test parsing fails gracefully with missing required field."""
        geo_data = {
            "accession": "GSE123456",
            # Missing title
            "gse": {"organism": "Homo sapiens", "sample_id": ["GSM1"]},
        }

        with pytest.raises(KeyError):
            parse_geo_metadata(geo_data)

    def test_parse_geo_detects_study_type(self):
        """Test parsing detects study type from GEO data."""
        geo_data = {
            "accession": "GSE123456",
            "title": "Test Study",
            "gse": {
                "organism": "Homo sapiens",
                "type": ["Expression profiling by array"],
                "sample_id": ["GSM1"],
            },
        }

        study = parse_geo_metadata(geo_data)

        assert study.study_type == "microarray"

    def test_parse_geo_rna_seq_type(self):
        """Test parsing detects RNA-seq study type."""
        geo_data = {
            "accession": "GSE123456",
            "title": "Test Study",
            "gse": {
                "organism": "Homo sapiens",
                "type": ["Expression profiling by high throughput sequencing"],
                "sample_id": ["GSM1"],
            },
        }

        study = parse_geo_metadata(geo_data)

        assert study.study_type == "RNA-seq"


class TestUpsertStudies:
    """Tests for study upsert functionality."""

    def test_upsert_studies_calls_database(self):
        """Test that upsert_studies calls the database."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{"id": 1}]
        mock_client.table.return_value.upsert.return_value.execute.return_value = mock_response

        studies = [
            Study(
                geo_accession="GSE123456",
                title="Test Study",
                organism="Homo sapiens",
                sample_count=50,
            )
        ]

        with patch("sage.data_loader.get_supabase_client", return_value=mock_client):
            result = upsert_studies(studies)

            assert result is not None
            mock_client.table.assert_called_with("studies")

    def test_upsert_empty_studies_list(self):
        """Test upserting empty list returns empty result."""
        mock_client = Mock()

        with patch("sage.data_loader.get_supabase_client", return_value=mock_client):
            result = upsert_studies([])

            assert result == []
            mock_client.table.assert_not_called()

    def test_upsert_converts_study_to_dict(self):
        """Test that Study objects are converted to dicts for upsert."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{"id": 1, "geo_accession": "GSE123456"}]
        mock_client.table.return_value.upsert.return_value.execute.return_value = mock_response

        study = Study(
            geo_accession="GSE123456",
            title="Test Study",
            organism="Homo sapiens",
            sample_count=50,
            has_sex_metadata=True,
            sex_metadata_completeness=0.75,
        )

        with patch("sage.data_loader.get_supabase_client", return_value=mock_client):
            upsert_studies([study])

            # Verify upsert was called with dict
            call_args = mock_client.table.return_value.upsert.call_args
            assert call_args is not None
