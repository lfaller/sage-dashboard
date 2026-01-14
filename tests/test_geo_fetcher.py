"""Tests for GEO fetcher module."""

import pytest
import time
from unittest.mock import patch
from sage.geo_fetcher import (
    GEOFetcher,
    RateLimiter,
    detect_sex_metadata_from_gse,
)
from tests.fixtures.geo_responses import (
    MOCK_GSE_RNA_SEQ_WITH_SEX,
    MOCK_GSE_RNA_SEQ_NO_SEX,
    MOCK_GSE_MICROARRAY_WITH_SEX,
    MOCK_GSE_MOUSE,
    MOCK_GSE_MINIMAL,
    MOCK_GSE_EMPTY_SAMPLES,
)


# ============================================================================
# Test Rate Limiter
# ============================================================================


class TestRateLimiter:
    """Test RateLimiter class for NCBI API compliance."""

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initializes with correct delay."""
        limiter = RateLimiter(requests_per_second=2.0)
        assert limiter.delay == 0.5  # 1.0 / 2.0

    def test_rate_limiter_enforces_minimum_delay(self):
        """Test RateLimiter enforces minimum delay between requests."""
        limiter = RateLimiter(requests_per_second=5.0)

        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start

        # Should take at least 0.2 seconds (1/5 second between requests)
        assert elapsed >= 0.19  # Allow small timing variance

    def test_rate_limiter_with_conservative_rate(self):
        """Test default conservative rate (2 req/sec)."""
        limiter = RateLimiter(requests_per_second=2.0)
        assert limiter.delay == 0.5

    def test_rate_limiter_with_high_rate(self):
        """Test rate limiter with high rate (for API key users)."""
        limiter = RateLimiter(requests_per_second=8.0)
        assert limiter.delay == pytest.approx(0.125, abs=0.001)

    def test_rate_limiter_first_call_no_delay(self):
        """Test first call doesn't add unnecessary delay."""
        limiter = RateLimiter(requests_per_second=1.0)

        start = time.time()
        limiter.wait()
        elapsed = time.time() - start

        # First call should be nearly instant
        assert elapsed < 0.1


# ============================================================================
# Test Sex Metadata Detection
# ============================================================================


class TestSexMetadataDetection:
    """Test sex metadata detection from GSE objects."""

    def test_detect_sex_metadata_clear_pattern_rna_seq(self):
        """Test detection of clear M/F pattern in sample names."""
        has_sex, completeness = detect_sex_metadata_from_gse(MOCK_GSE_RNA_SEQ_WITH_SEX)

        assert has_sex is True
        assert completeness > 0.8  # Should be high (clear pattern)

    def test_detect_sex_metadata_no_pattern(self):
        """Test detection when no sex pattern in sample names."""
        has_sex, completeness = detect_sex_metadata_from_gse(MOCK_GSE_RNA_SEQ_NO_SEX)

        assert has_sex is False
        assert completeness == 0.0

    def test_detect_sex_metadata_partial_pattern(self):
        """Test detection with partial sex metadata."""
        has_sex, completeness = detect_sex_metadata_from_gse(MOCK_GSE_MICROARRAY_WITH_SEX)

        assert has_sex is True
        assert 0.0 < completeness <= 1.0

    def test_detect_sex_metadata_empty_samples(self):
        """Test handling of GSE with no samples."""
        has_sex, completeness = detect_sex_metadata_from_gse(MOCK_GSE_EMPTY_SAMPLES)

        assert has_sex is False
        assert completeness == 0.0

    def test_detect_sex_metadata_mouse_study(self):
        """Test detection works on non-human organisms."""
        has_sex, completeness = detect_sex_metadata_from_gse(MOCK_GSE_MOUSE)

        assert has_sex is True  # M/F pattern present
        assert completeness > 0.0


# ============================================================================
# Test GEO Fetcher
# ============================================================================


class TestGEOFetcher:
    """Test GEOFetcher class."""

    def test_geo_fetcher_initialization(self):
        """Test GEOFetcher initializes correctly."""
        fetcher = GEOFetcher(rate_limit=2.0)

        assert fetcher.rate_limiter is not None
        assert fetcher.use_cache is True

    def test_geo_fetcher_with_custom_rate_limit(self):
        """Test GEOFetcher with custom rate limit."""
        fetcher = GEOFetcher(rate_limit=5.0)
        assert fetcher.rate_limiter.delay == pytest.approx(0.2, abs=0.001)

    @patch("GEOparse.get_GEO")
    def test_fetch_study_returns_study_object(self, mock_get_geo):
        """Test fetching single study returns Study object."""
        mock_get_geo.return_value = MOCK_GSE_RNA_SEQ_WITH_SEX

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE123456")

        assert study is not None
        assert study.geo_accession == "GSE123456"
        assert study.title == "RNA-seq of breast cancer samples with sex differences"
        assert study.organism == "Homo sapiens"

    @patch("GEOparse.get_GEO")
    def test_fetch_study_detects_rna_seq_type(self, mock_get_geo):
        """Test study type detection for RNA-seq."""
        mock_get_geo.return_value = MOCK_GSE_RNA_SEQ_WITH_SEX

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE123456")

        assert study.study_type == "RNA-seq"

    @patch("GEOparse.get_GEO")
    def test_fetch_study_detects_microarray_type(self, mock_get_geo):
        """Test study type detection for microarray."""
        mock_get_geo.return_value = MOCK_GSE_MICROARRAY_WITH_SEX

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE789012")

        assert study.study_type == "microarray"

    @patch("GEOparse.get_GEO")
    def test_fetch_study_detects_sex_metadata(self, mock_get_geo):
        """Test sex metadata detection during fetch."""
        mock_get_geo.return_value = MOCK_GSE_RNA_SEQ_WITH_SEX

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE123456")

        assert study.has_sex_metadata is True
        assert study.sex_metadata_completeness > 0.0

    @patch("GEOparse.get_GEO")
    def test_fetch_study_handles_missing_optional_fields(self, mock_get_geo):
        """Test fetching study with missing optional fields."""
        mock_get_geo.return_value = MOCK_GSE_MINIMAL

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE567890")

        assert study is not None
        assert study.geo_accession == "GSE567890"
        assert study.title == "Simple study"
        # Optional fields should be None or default values
        assert study.summary is None

    @patch("GEOparse.get_GEO")
    def test_fetch_study_handles_network_error(self, mock_get_geo):
        """Test graceful handling of network errors."""
        mock_get_geo.side_effect = ConnectionError("Network error")

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE999999")

        assert study is None

    @patch("GEOparse.get_GEO")
    def test_fetch_study_retries_on_failure(self, mock_get_geo):
        """Test that fetcher retries after failure."""
        mock_get_geo.side_effect = [
            Exception("First attempt"),
            Exception("Second attempt"),
            MOCK_GSE_RNA_SEQ_WITH_SEX,
        ]

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE123456", retry_count=3)

        # Should succeed on third attempt
        assert study is not None
        assert study.geo_accession == "GSE123456"
        assert mock_get_geo.call_count == 3

    @patch("GEOparse.get_GEO")
    def test_fetch_study_fails_after_retries_exhausted(self, mock_get_geo):
        """Test that fetcher fails after retries exhausted."""
        mock_get_geo.side_effect = Exception("Always fails")

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE999999", retry_count=2)

        assert study is None
        assert mock_get_geo.call_count == 2

    @patch("GEOparse.get_GEO")
    def test_fetch_multiple_studies(self, mock_get_geo):
        """Test fetching multiple studies."""
        mock_get_geo.side_effect = [
            MOCK_GSE_RNA_SEQ_WITH_SEX,
            MOCK_GSE_MICROARRAY_WITH_SEX,
        ]

        fetcher = GEOFetcher()
        studies = fetcher.fetch_multiple_studies(["GSE123456", "GSE789012"])

        assert len(studies) == 2
        assert studies[0].geo_accession == "GSE123456"
        assert studies[1].geo_accession == "GSE789012"

    @patch("GEOparse.get_GEO")
    def test_fetch_multiple_studies_skips_existing(self, mock_get_geo):
        """Test that fetcher skips existing studies."""
        mock_get_geo.return_value = MOCK_GSE_RNA_SEQ_WITH_SEX

        fetcher = GEOFetcher()
        existing = {"GSE123456"}
        studies = fetcher.fetch_multiple_studies(["GSE123456", "GSE234567"], skip_existing=existing)

        # Should only fetch GSE234567 (one accession)
        assert len(studies) == 1
        assert mock_get_geo.call_count == 1

    @patch("GEOparse.get_GEO")
    def test_fetch_multiple_studies_handles_failures(self, mock_get_geo):
        """Test that multiple fetch continues despite individual failures."""
        mock_get_geo.side_effect = [
            MOCK_GSE_RNA_SEQ_WITH_SEX,
            Exception("Fetch failed"),
            MOCK_GSE_MICROARRAY_WITH_SEX,
        ]

        fetcher = GEOFetcher()
        studies = fetcher.fetch_multiple_studies(["GSE123456", "GSE999999", "GSE789012"])

        # Should have 2 successful studies despite 1 failure
        assert len(studies) == 2
        assert studies[0].geo_accession == "GSE123456"
        assert studies[1].geo_accession == "GSE789012"

    @patch("GEOparse.get_GEO")
    def test_fetch_study_sample_count(self, mock_get_geo):
        """Test that sample count is correctly derived from GSE."""
        mock_get_geo.return_value = MOCK_GSE_RNA_SEQ_WITH_SEX

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE123456")

        # MOCK_GSE_RNA_SEQ_WITH_SEX has 6 GSM samples
        assert study.sample_count == 6

    @patch("GEOparse.get_GEO")
    def test_fetch_study_with_non_human_organism(self, mock_get_geo):
        """Test fetching non-human organism study."""
        mock_get_geo.return_value = MOCK_GSE_MOUSE

        fetcher = GEOFetcher()
        study = fetcher.fetch_study("GSE456789")

        assert study.organism == "Mus musculus"
        assert "mouse" in study.title.lower()
