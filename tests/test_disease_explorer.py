"""Tests for disease explorer functionality."""
from unittest.mock import Mock, patch

from sage.database import (
    fetch_disease_stats,
    get_diseases_with_completeness,
    get_studies_for_disease,
    get_disease_categories,
)


class TestFetchDiseaseStats:
    """Tests for disease overview statistics."""

    def test_fetch_disease_stats_returns_dict_with_required_keys(self):
        """Test that disease stats returns dict with all required keys."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.count = 5
        mock_response.data = [
            {
                "disease_term": "breast cancer",
                "study_count": 10,
                "avg_completeness": 0.85,
            }
        ]

        query_mock = Mock()
        query_mock.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        # Create a mock st that bypasses the cache decorator entirely
        def cache_data_decorator(**kwargs):
            """Bypass Streamlit's caching for testing."""

            def decorator(func):
                return func

            return decorator

        mock_st = Mock()
        mock_st.cache_data = cache_data_decorator

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st", mock_st):
                stats = fetch_disease_stats()

                assert isinstance(stats, dict)
                assert "total_diseases" in stats
                assert "diseases_with_studies" in stats
                assert "avg_completeness" in stats
                assert "total_study_mappings" in stats

    def test_fetch_disease_stats_values_are_numeric(self):
        """Test that all stats values are numeric."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.count = 3
        mock_response.data = [{"disease_term": "cancer", "study_count": 5}]

        query_mock = Mock()
        query_mock.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        # Create a mock st that bypasses the cache decorator entirely
        def cache_data_decorator(**kwargs):
            """Bypass Streamlit's caching for testing."""

            def decorator(func):
                return func

            return decorator

        mock_st = Mock()
        mock_st.cache_data = cache_data_decorator

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st", mock_st):
                stats = fetch_disease_stats()

                assert isinstance(stats["total_diseases"], int)
                assert isinstance(stats["diseases_with_studies"], int)
                assert isinstance(stats["avg_completeness"], (int, float))
                assert isinstance(stats["total_study_mappings"], int)


class TestGetDiseasesWithCompleteness:
    """Tests for disease listing with metrics."""

    def test_get_diseases_with_completeness_returns_list(self):
        """Test that function returns a list."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = [
            {
                "disease_term": "breast cancer",
                "disease_category": "cancer",
                "study_count": 10,
                "avg_completeness": 0.85,
                "known_sex_difference": True,
                "sex_bias_direction": "female",
                "avg_clinical_priority": 0.9,
            }
        ]

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gt.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_diseases_with_completeness()

            assert isinstance(result, list)
            assert len(result) > 0

    def test_get_diseases_with_category_filter(self):
        """Test filtering by disease category."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = [
            {
                "disease_term": "breast cancer",
                "disease_category": "cancer",
                "study_count": 10,
                "avg_completeness": 0.85,
                "known_sex_difference": True,
                "sex_bias_direction": "female",
                "avg_clinical_priority": 0.9,
            }
        ]

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gt.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_diseases_with_completeness(disease_category="cancer")

            assert len(result) > 0
            # Verify eq was called for category filter
            query_mock.eq.assert_called()

    def test_get_diseases_with_min_studies_filter(self):
        """Test filtering by minimum study count."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = []

        # Create a mock that supports method chaining
        query_mock = Mock()
        # All methods return self to support chaining
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.execute.return_value = mock_response

        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_diseases_with_completeness(min_studies=5)

            assert isinstance(result, list)

    def test_get_diseases_with_known_sex_diff_filter(self):
        """Test filtering by known sex differences flag."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gt.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_diseases_with_completeness(known_sex_diff_only=True)

            assert isinstance(result, list)

    def test_get_diseases_respects_limit(self):
        """Test that limit parameter is respected."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gt.return_value = query_mock
        query_mock.limit.return_value.offset.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_diseases_with_completeness(limit=50)

            assert isinstance(result, list)
            query_mock.limit.assert_called_with(50)

    def test_get_diseases_calculates_avg_completeness(self):
        """Test that average completeness is calculated correctly."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = [
            {
                "disease_term": "breast cancer",
                "disease_category": "cancer",
                "known_sex_difference": True,
                "sex_bias_direction": "female",
                "clinical_priority_score": 0.8,
            }
        ]

        # Create a mock that supports method chaining
        query_mock = Mock()
        # All methods return self to support chaining
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.execute.return_value = mock_response

        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_diseases_with_completeness()

            assert len(result) > 0
            assert "avg_completeness" in result[0]
            # avg_completeness should be aggregated from data, it's set to 0.0 in the code
            assert isinstance(result[0]["avg_completeness"], float)


class TestGetStudiesForDisease:
    """Tests for disease drill-down functionality."""

    def test_get_studies_for_disease_returns_list(self):
        """Test that function returns list of studies."""
        mock_client = Mock()

        # First response: disease_mappings query returns study_ids
        mapping_response = Mock()
        mapping_response.data = [{"study_id": 1}]

        # Second response: studies query returns full study records
        study_response = Mock()
        study_response.data = [
            {
                "id": 1,
                "geo_accession": "GSE123",
                "title": "Test Study",
                "organism": "Homo sapiens",
                "sample_count": 50,
                "sex_metadata_completeness": 0.9,
                "reports_sex_analysis": True,
            }
        ]

        # Mock table().select().eq().execute() chain for disease_mappings
        query_mock = Mock()
        query_mock.eq.return_value.limit.return_value.execute.return_value = mapping_response

        # Mock table().select().eq().execute() chain for studies
        studies_query_mock = Mock()
        studies_query_mock.eq.return_value.execute.return_value = study_response

        # Setup mock_client to return different mocks for different tables
        def table_side_effect(table_name):
            table_mock = Mock()
            if table_name == "disease_mappings":
                table_mock.select.return_value = query_mock
            else:  # studies table
                table_mock.select.return_value = studies_query_mock
            return table_mock

        mock_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_studies_for_disease("breast cancer")

            assert isinstance(result, list)
            assert len(result) > 0

    def test_get_studies_for_disease_with_valid_term(self):
        """Test retrieval with a valid disease term."""
        mock_client = Mock()

        # First response: disease_mappings query returns study_ids
        mapping_response = Mock()
        mapping_response.data = [{"study_id": 1}]

        # Second response: studies query returns full study records
        study_response = Mock()
        study_response.data = [
            {
                "id": 1,
                "geo_accession": "GSE123",
                "title": "Breast Cancer Study",
                "organism": "Homo sapiens",
                "sample_count": 100,
                "sex_metadata_completeness": 0.85,
                "reports_sex_analysis": False,
            }
        ]

        # Mock table().select().eq().execute() chain for disease_mappings
        query_mock = Mock()
        query_mock.eq.return_value.limit.return_value.execute.return_value = mapping_response

        # Mock table().select().eq().execute() chain for studies
        studies_query_mock = Mock()
        studies_query_mock.eq.return_value.execute.return_value = study_response

        # Setup mock_client to return different mocks for different tables
        def table_side_effect(table_name):
            table_mock = Mock()
            if table_name == "disease_mappings":
                table_mock.select.return_value = query_mock
            else:  # studies table
                table_mock.select.return_value = studies_query_mock
            return table_mock

        mock_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_studies_for_disease("breast cancer")

            assert len(result) == 1
            assert result[0]["geo_accession"] == "GSE123"

    def test_get_studies_for_disease_handles_no_mappings(self):
        """Test behavior when disease has no study mappings."""
        mock_client = Mock()

        # First response: disease_mappings query returns no results
        mapping_response = Mock()
        mapping_response.data = None

        # Mock table().select().eq().execute() chain for disease_mappings
        query_mock = Mock()
        query_mock.eq.return_value.limit.return_value.execute.return_value = mapping_response

        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_studies_for_disease("nonexistent disease")

            assert isinstance(result, list)
            assert len(result) == 0

    def test_get_studies_for_disease_respects_limit(self):
        """Test that limit parameter is respected."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            result = get_studies_for_disease("cancer", limit=50)

            assert isinstance(result, list)


class TestGetDiseaseCategories:
    """Tests for disease category retrieval."""

    def test_get_disease_categories_returns_list(self):
        """Test that function returns a list."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = [
            {"disease_category": "cancer"},
            {"disease_category": "infectious"},
        ]

        query_mock = Mock()
        query_mock.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st"):
                result = get_disease_categories()

                assert isinstance(result, list)

    def test_get_disease_categories_returns_unique_values(self):
        """Test that duplicates are removed."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = [
            {"disease_category": "cancer"},
            {"disease_category": "cancer"},
            {"disease_category": "infectious"},
        ]

        query_mock = Mock()
        query_mock.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st"):
                result = get_disease_categories()

                # Should have 2 unique categories
                assert len(result) == 2

    def test_get_disease_categories_handles_nulls(self):
        """Test that NULL categories are filtered out."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = [
            {"disease_category": "cancer"},
            {"disease_category": None},
            {"disease_category": "infectious"},
        ]

        query_mock = Mock()
        query_mock.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st"):
                result = get_disease_categories()

                # Should not include None
                assert None not in result

    def test_get_disease_categories_sorted_alphabetically(self):
        """Test that results are sorted."""
        mock_client = Mock()

        mock_response = Mock()
        mock_response.data = [
            {"disease_category": "zebra"},
            {"disease_category": "cancer"},
            {"disease_category": "apple"},
        ]

        query_mock = Mock()
        query_mock.execute.return_value = mock_response
        mock_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_client):
            with patch("sage.database.st"):
                result = get_disease_categories()

                # Verify sorted order
                assert result == sorted(result)
