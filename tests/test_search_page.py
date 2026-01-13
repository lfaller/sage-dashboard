"""Tests for study search functionality."""
from unittest.mock import Mock, patch

from sage.database import search_studies_advanced, get_filter_options, get_study_by_accession


class TestSearchStudiesAdvanced:
    """Tests for advanced search functionality."""

    def test_search_with_text_query(self):
        """Test text search across title, summary, and accession."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 5
        mock_response.data = [
            {
                "id": 1,
                "geo_accession": "GSE123001",
                "title": "Breast cancer study",
                "organism": "Homo sapiens",
                "study_type": "RNA-seq",
                "sample_count": 100,
                "has_sex_metadata": True,
                "sex_metadata_completeness": 0.92,
            }
        ]

        query_mock = Mock()
        query_mock.ilike.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = search_studies_advanced(search="breast")

            assert "results" in result
            assert "total" in result
            assert result["total"] == 5
            assert len(result["results"]) == 1

    def test_search_with_organism_filter(self):
        """Test filtering by organism."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 3
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = search_studies_advanced(organism="Homo sapiens")

            assert result["total"] == 3
            # Verify eq was called for organism filter
            mock_client.table.return_value.select.return_value.eq.assert_called()

    def test_search_with_study_type_filter(self):
        """Test filtering by study type."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 10
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = search_studies_advanced(study_type="RNA-seq")

            assert result["total"] == 10

    def test_search_with_sex_metadata_filter(self):
        """Test filtering by sex metadata availability."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 7
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = search_studies_advanced(has_sex_metadata=True)

            assert result["total"] == 7

    def test_search_with_multiple_filters(self):
        """Test applying multiple filters together (AND logic)."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 2
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = search_studies_advanced(
                organism="Homo sapiens",
                study_type="RNA-seq",
                has_sex_metadata=True,
            )

            assert result["total"] == 2

    def test_search_with_pagination(self):
        """Test pagination with limit and offset."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 150
        mock_response.data = []

        query_mock = Mock()
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = search_studies_advanced(limit=50, offset=100)

            assert result["total"] == 150
            # Verify limit and offset were called
            query_mock.limit.assert_called_with(50)
            query_mock.offset.assert_called_with(100)

    def test_search_with_empty_results(self):
        """Test search with no matching results."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 0
        mock_response.data = None

        query_mock = Mock()
        query_mock.ilike.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = search_studies_advanced(search="nonexistent")

            assert result["total"] == 0
            assert result["results"] == []


class TestGetFilterOptions:
    """Tests for getting unique filter values."""

    def test_get_filter_options_returns_dict(self):
        """Test that filter options returns dict with expected keys."""
        mock_client = Mock()

        organisms_response = Mock()
        organisms_response.data = [{"organism": "Homo sapiens"}, {"organism": "Mus musculus"}]

        study_types_response = Mock()
        study_types_response.data = [{"study_type": "RNA-seq"}, {"study_type": "microarray"}]

        platforms_response = Mock()
        platforms_response.data = [{"platform": "Illumina HiSeq"}, {"platform": "Affymetrix"}]

        def make_query_mock(response):
            query = Mock()
            query.execute.return_value = response
            return query

        table_mock = Mock()
        table_mock.select.side_effect = [
            make_query_mock(organisms_response),
            make_query_mock(study_types_response),
            make_query_mock(platforms_response),
        ]
        mock_client.table.return_value = table_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st"):
                options = get_filter_options()

                assert "organisms" in options
                assert "study_types" in options
                assert "platforms" in options

    def test_get_filter_options_returns_lists(self):
        """Test that filter options returns lists of unique values."""
        mock_client = Mock()

        organisms_response = Mock()
        organisms_response.data = [{"organism": "Homo sapiens"}, {"organism": "Mus musculus"}]

        study_types_response = Mock()
        study_types_response.data = [{"study_type": "RNA-seq"}, {"study_type": "microarray"}]

        platforms_response = Mock()
        platforms_response.data = [{"platform": "Illumina HiSeq"}, {"platform": "Affymetrix"}]

        def make_query_mock(response):
            query = Mock()
            query.execute.return_value = response
            return query

        table_mock = Mock()
        table_mock.select.side_effect = [
            make_query_mock(organisms_response),
            make_query_mock(study_types_response),
            make_query_mock(platforms_response),
        ]
        mock_client.table.return_value = table_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st"):
                options = get_filter_options()

                assert isinstance(options["organisms"], list)
                assert isinstance(options["study_types"], list)
                assert isinstance(options["platforms"], list)
                assert len(options["organisms"]) > 0
                assert len(options["study_types"]) > 0


class TestGetStudyByAccession:
    """Tests for study lookup by accession."""

    def test_get_study_by_accession_returns_study(self):
        """Test retrieving a study by its GEO accession."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            {
                "id": 1,
                "geo_accession": "GSE123001",
                "title": "Test Study",
                "organism": "Homo sapiens",
                "study_type": "RNA-seq",
                "sample_count": 100,
                "has_sex_metadata": True,
                "sex_metadata_completeness": 0.92,
                "summary": "A test study",
                "platform": "Illumina HiSeq",
                "pubmed_id": "12345678",
            }
        ]

        query_mock = Mock()
        query_mock.eq.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            study = get_study_by_accession("GSE123001")

            assert study is not None
            assert study["geo_accession"] == "GSE123001"
            assert study["title"] == "Test Study"

    def test_get_study_by_accession_returns_none_when_not_found(self):
        """Test retrieving non-existent study returns None."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = None

        query_mock = Mock()
        query_mock.eq.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            study = get_study_by_accession("GSE000000")

            assert study is None

    def test_get_study_by_accession_uses_exact_match(self):
        """Test that accession lookup uses exact match, not partial."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            get_study_by_accession("GSE123")

            # Verify eq was called for exact match
            query_mock.eq.assert_called_with("geo_accession", "GSE123")
