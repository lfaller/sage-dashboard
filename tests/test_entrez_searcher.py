"""Tests for Entrez API search functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from sage.entrez_searcher import EntrezSearcher


class TestEntrezSearcher:
    def test_search_recent_studies_returns_list(self):
        """Test that search returns list of accessions."""
        with patch("sage.entrez_searcher.Entrez") as mock_entrez:
            # Mock Entrez.esearch response
            mock_search_handle = MagicMock()
            mock_entrez.esearch.return_value = mock_search_handle
            mock_entrez.read.return_value = {
                "IdList": ["12345", "67890"],
                "Count": "2",
            }

            # Mock _convert_gds_to_gse
            searcher = EntrezSearcher(email="test@example.com")
            searcher._convert_gds_to_gse = Mock(return_value=["GSE12345", "GSE67890"])

            result = searcher.search_recent_studies(
                organism="Homo sapiens",
                study_type="RNA-seq",
                years_back=5,
                max_results=100,
            )

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0].startswith("GSE")

    def test_search_builds_correct_query(self):
        """Test that query is built with correct syntax."""
        with patch("sage.entrez_searcher.Entrez") as mock_entrez:
            mock_search_handle = MagicMock()
            mock_entrez.esearch.return_value = mock_search_handle
            mock_entrez.read.return_value = {"IdList": []}

            searcher = EntrezSearcher(email="test@example.com")
            searcher.search_recent_studies(
                organism="Mus musculus",
                study_type="RNA-seq",
                years_back=2,
            )

            # Verify query structure
            call_args = mock_entrez.esearch.call_args
            query = call_args[1]["term"]

            assert "Mus musculus" in query
            assert "Expression profiling by high throughput sequencing" in query
            assert "PDAT" in query  # Publication date filter

    def test_convert_gds_to_gse(self):
        """Test GDS ID to GSE accession conversion."""
        with patch("sage.entrez_searcher.Entrez") as mock_entrez:
            # Mock Entrez.esummary response
            mock_summary_handle = MagicMock()
            mock_entrez.esummary.return_value = mock_summary_handle
            mock_entrez.read.return_value = [{"Accession": "GSE123456", "Summary": "Study summary"}]

            searcher = EntrezSearcher(email="test@example.com")
            result = searcher._convert_gds_to_gse(["12345"])

            assert result == ["GSE123456"]

    def test_search_handles_empty_results(self):
        """Test that search handles no results gracefully."""
        with patch("sage.entrez_searcher.Entrez") as mock_entrez:
            mock_search_handle = MagicMock()
            mock_entrez.esearch.return_value = mock_search_handle
            mock_entrez.read.return_value = {"IdList": []}

            searcher = EntrezSearcher(email="test@example.com")
            result = searcher.search_recent_studies()

            assert result == []

    def test_invalid_organism_raises_error(self):
        """Test that invalid organism raises ValueError."""
        with patch("sage.entrez_searcher.Entrez"):
            searcher = EntrezSearcher(email="test@example.com")

            with pytest.raises(ValueError):
                searcher.search_recent_studies(organism="Invalid organism")

    def test_invalid_study_type_raises_error(self):
        """Test that invalid study type raises ValueError."""
        with patch("sage.entrez_searcher.Entrez"):
            searcher = EntrezSearcher(email="test@example.com")

            with pytest.raises(ValueError):
                searcher.search_recent_studies(study_type="invalid_type")

    def test_convert_gds_skips_non_gse_accessions(self):
        """Test that GDS to GSE conversion skips non-GSE accessions."""
        with patch("sage.entrez_searcher.Entrez") as mock_entrez:
            # Mock mixed accession types
            def side_effect(*args, **kwargs):
                handle = MagicMock()
                return handle

            mock_entrez.esummary.side_effect = side_effect

            # Mock reads for multiple calls
            mock_results = [
                [{"Accession": "GSE123456"}],
                [{"Accession": "GDS789"}],
                [{"Accession": "GSE234567"}],
            ]

            with patch("sage.entrez_searcher.Entrez.read") as mock_read:
                mock_read.side_effect = mock_results

                searcher = EntrezSearcher(email="test@example.com")
                result = searcher._convert_gds_to_gse(["1", "2", "3"])

                # Should only include GSE accessions
                assert len(result) == 2
                assert "GSE123456" in result
                assert "GSE234567" in result
                assert "GDS789" not in result

    def test_convert_gds_handles_conversion_errors(self):
        """Test that GDS to GSE conversion handles errors gracefully."""
        with patch("sage.entrez_searcher.Entrez") as mock_entrez:
            # First call fails, second succeeds
            def side_effect(*args, **kwargs):
                handle = MagicMock()
                return handle

            mock_entrez.esummary.side_effect = side_effect

            mock_results = [
                Exception("Network error"),
                [{"Accession": "GSE123456"}],
            ]

            with patch("sage.entrez_searcher.Entrez.read") as mock_read:
                mock_read.side_effect = mock_results

                searcher = EntrezSearcher(email="test@example.com")
                result = searcher._convert_gds_to_gse(["1", "2"])

                # Should skip failed conversion and continue
                assert len(result) == 1
                assert "GSE123456" in result
