"""Tests for Supabase database module."""
import pytest
from unittest.mock import Mock, patch

from sage.database import (
    fetch_overview_stats,
    search_studies,
    StudyFilter,
)


@pytest.fixture
def mock_supabase_client():
    """Fixture providing a mocked Supabase client."""
    return Mock()


class TestOverviewStats:
    """Tests for overview statistics fetching."""

    def test_fetch_overview_stats_returns_dict_with_required_keys(self, mock_supabase_client):
        """Test that overview stats returns dict with all required keys."""
        mock_response = Mock()
        mock_response.count = 100
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = (
            mock_response
        )
        mock_supabase_client.table.return_value.select.return_value.gt.return_value.execute.return_value = (
            mock_response
        )
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            stats = fetch_overview_stats()

            assert isinstance(stats, dict)
            assert "total_studies" in stats
            assert "with_sex_metadata" in stats
            assert "sex_inferrable" in stats
            assert "with_sex_analysis" in stats

    def test_fetch_overview_stats_values_are_integers(self, mock_supabase_client):
        """Test that all stats values are integers."""
        mock_response = Mock()
        mock_response.count = 100
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = (
            mock_response
        )
        mock_supabase_client.table.return_value.select.return_value.gt.return_value.execute.return_value = (
            mock_response
        )
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            stats = fetch_overview_stats()

            for key, value in stats.items():
                assert isinstance(value, int), f"{key} should be int, got {type(value)}"


class TestStudyFilter:
    """Tests for StudyFilter helper class."""

    def test_study_filter_initialization(self):
        """Test StudyFilter can be initialized."""
        filter_obj = StudyFilter()
        assert filter_obj.organism is None
        assert filter_obj.has_sex_metadata is None
        assert filter_obj.sex_inferrable is None

    def test_study_filter_with_organism(self):
        """Test StudyFilter with organism filter."""
        filter_obj = StudyFilter(organism="Homo sapiens")
        assert filter_obj.organism == "Homo sapiens"

    def test_study_filter_with_multiple_filters(self):
        """Test StudyFilter with multiple filters."""
        filter_obj = StudyFilter(organism="Homo sapiens", has_sex_metadata=True, limit=50)
        assert filter_obj.organism == "Homo sapiens"
        assert filter_obj.has_sex_metadata is True
        assert filter_obj.limit == 50


class TestSearchStudies:
    """Tests for study search functionality."""

    def test_search_studies_returns_list(self, mock_supabase_client):
        """Test that search_studies returns a list."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies()
            assert isinstance(results, list)

    def test_search_studies_with_organism_filter(self, mock_supabase_client):
        """Test search_studies with organism filter."""
        mock_response = Mock()
        mock_response.data = [{"id": 1, "organism": "Homo sapiens"}]

        query_mock = Mock()
        query_mock.eq.return_value.limit.return_value.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies(organism="Homo sapiens")

            assert len(results) == 1
            assert results[0]["organism"] == "Homo sapiens"

    def test_search_studies_respects_limit(self, mock_supabase_client):
        """Test that search_studies respects limit parameter."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            search_studies(limit=25)

            # Verify limit was called
            mock_supabase_client.table.return_value.select.return_value.limit.assert_called()

    def test_search_studies_handles_empty_results(self, mock_supabase_client):
        """Test search_studies handles empty results gracefully."""
        mock_response = Mock()
        mock_response.data = None
        mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies()

            assert results == []

    def test_search_studies_with_multiple_filters(self, mock_supabase_client):
        """Test search_studies with multiple filters applied."""
        mock_response = Mock()
        mock_response.data = [{"id": 1, "organism": "Homo sapiens", "has_sex_metadata": True}]

        query_mock = Mock()
        eq_mock = Mock()
        eq_mock.eq.return_value.limit.return_value.execute.return_value = mock_response
        query_mock.eq.return_value = eq_mock
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies(organism="Homo sapiens", has_sex_metadata=True)

            assert len(results) == 1
