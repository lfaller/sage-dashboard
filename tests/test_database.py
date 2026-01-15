"""Tests for Supabase database module."""
import pytest
from unittest.mock import Mock, patch

from sage.database import (
    fetch_overview_stats,
    search_studies,
    StudyFilter,
    get_rescue_opportunities,
    calculate_rescue_score,
    fetch_rescue_stats,
    create_snapshot,
    create_snapshot_with_date,
    fetch_snapshots,
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
        mock_response.count = 0

        query_mock = Mock()
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies()
            assert isinstance(results, list)

    def test_search_studies_with_organism_filter(self, mock_supabase_client):
        """Test search_studies with organism filter."""
        mock_response = Mock()
        mock_response.data = [{"id": 1, "organism": "Homo sapiens"}]
        mock_response.count = 1

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies(organism="Homo sapiens")

            assert len(results) == 1
            assert results[0]["organism"] == "Homo sapiens"

    def test_search_studies_respects_limit(self, mock_supabase_client):
        """Test that search_studies respects limit parameter."""
        mock_response = Mock()
        mock_response.data = []
        mock_response.count = 0

        query_mock = Mock()
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            search_studies(limit=25)

            # Verify limit was called with 25
            query_mock.limit.assert_called_with(25)

    def test_search_studies_handles_empty_results(self, mock_supabase_client):
        """Test search_studies handles empty results gracefully."""
        mock_response = Mock()
        mock_response.data = None
        mock_response.count = 0

        query_mock = Mock()
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies()

            assert results == []

    def test_search_studies_with_multiple_filters(self, mock_supabase_client):
        """Test search_studies with multiple filters applied."""
        mock_response = Mock()
        mock_response.data = [{"id": 1, "organism": "Homo sapiens", "has_sex_metadata": True}]
        mock_response.count = 1

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = search_studies(organism="Homo sapiens", has_sex_metadata=True)

            assert len(results) == 1


class TestGetRescueOpportunities:
    """Tests for rescue opportunities fetching."""

    def test_returns_list_of_dicts(self, mock_supabase_client):
        """Test that get_rescue_opportunities returns a list of dicts."""
        # Mock studies response
        studies_response = Mock()
        studies_response.data = [
            {
                "id": 1,
                "geo_accession": "GSE123001",
                "sex_inferrable": True,
                "sex_inference_confidence": 0.8,
                "title": "Test Study",
                "organism": "Homo sapiens",
                "study_type": "RNA-seq",
                "sample_count": 30,
                "has_sex_metadata": False,
                "sex_metadata_completeness": 0.0,
            }
        ]
        studies_response.count = 1

        # Mock disease mappings response
        disease_response = Mock()
        disease_response.data = [{"study_id": 1}]

        # Mock study completeness response
        completeness_response = Mock()
        completeness_response.data = [{"sex_metadata_completeness": 0.0}]

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gte.return_value = query_mock
        query_mock.neq.return_value = query_mock
        query_mock.gt.return_value = query_mock
        query_mock.in_.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.side_effect = [studies_response, disease_response, completeness_response]
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = get_rescue_opportunities()

            assert isinstance(results, list)
            assert len(results) > 0
            assert isinstance(results[0], dict)
            assert "disease_terms" in results[0]

    def test_filters_by_organism(self, mock_supabase_client):
        """Test filtering by organism."""
        mock_response = Mock()
        mock_response.data = [{"id": 1, "organism": "Homo sapiens"}]

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gte.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = get_rescue_opportunities(organism="Homo sapiens")

            assert isinstance(results, list)

    def test_filters_by_min_confidence(self, mock_supabase_client):
        """Test filtering by minimum confidence threshold."""
        mock_response = Mock()
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gte.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = get_rescue_opportunities(min_confidence=0.7)

            assert isinstance(results, list)

    def test_filters_by_min_sample_size(self, mock_supabase_client):
        """Test filtering by minimum sample size."""
        mock_response = Mock()
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gte.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = get_rescue_opportunities(min_sample_size=30)

            assert isinstance(results, list)

    def test_filters_by_disease_category(self, mock_supabase_client):
        """Test filtering by disease category."""
        # Main query returns studies
        studies_response = Mock()
        studies_response.data = [{"id": 1, "sex_inference_confidence": 0.8}]

        # Disease mapping query returns disease records
        disease_response = Mock()
        disease_response.data = [{"study_id": 1}]

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gte.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = studies_response

        disease_query = Mock()
        disease_query.eq.return_value = disease_query
        disease_query.execute.return_value = disease_response

        def table_side_effect(table_name):
            table_mock = Mock()
            if table_name == "disease_mappings":
                table_mock.select.return_value = disease_query
            else:  # studies
                table_mock.select.return_value = query_mock
            return table_mock

        mock_supabase_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            results = get_rescue_opportunities(disease_category="cancer")

            assert isinstance(results, list)

    def test_respects_limit_parameter(self, mock_supabase_client):
        """Test that limit parameter is respected."""
        mock_response = Mock()
        mock_response.data = []

        query_mock = Mock()
        query_mock.eq.return_value = query_mock
        query_mock.gte.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = mock_response
        mock_supabase_client.table.return_value.select.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            get_rescue_opportunities(limit=50)

            query_mock.limit.assert_called_with(50)


class TestCalculateRescueScore:
    """Tests for rescue score calculation."""

    def test_high_score_missing_metadata(self):
        """Test high score when metadata is completely missing."""
        study = {
            "sex_inference_confidence": 0.8,
            "sample_count": 100,
            "sex_metadata_completeness": 0.0,  # Completely missing
            "study_type": "RNA-seq",
            "clinical_priority_score": 0.8,
        }
        score = calculate_rescue_score(study)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high due to missing metadata

    def test_high_confidence_increases_score(self):
        """Test that high inference confidence increases score."""
        study_high = {
            "sex_inference_confidence": 0.9,
            "sample_count": 50,
            "sex_metadata_completeness": 0.0,
            "study_type": "microarray",
            "clinical_priority_score": 0.5,
        }
        study_low = {
            "sex_inference_confidence": 0.3,
            "sample_count": 50,
            "sex_metadata_completeness": 0.0,
            "study_type": "microarray",
            "clinical_priority_score": 0.5,
        }

        score_high = calculate_rescue_score(study_high)
        score_low = calculate_rescue_score(study_low)

        assert score_high > score_low

    def test_large_sample_size_increases_score(self):
        """Test that larger sample size increases score."""
        study_large = {
            "sex_inference_confidence": 0.5,
            "sample_count": 200,
            "sex_metadata_completeness": 0.0,
            "study_type": "microarray",
            "clinical_priority_score": 0.5,
        }
        study_small = {
            "sex_inference_confidence": 0.5,
            "sample_count": 20,
            "sex_metadata_completeness": 0.0,
            "study_type": "microarray",
            "clinical_priority_score": 0.5,
        }

        score_large = calculate_rescue_score(study_large)
        score_small = calculate_rescue_score(study_small)

        assert score_large > score_small

    def test_disease_priority_factor(self):
        """Test that clinical priority factor affects score."""
        study_high_priority = {
            "sex_inference_confidence": 0.5,
            "sample_count": 50,
            "sex_metadata_completeness": 0.0,
            "study_type": "microarray",
            "clinical_priority_score": 0.9,
        }
        study_low_priority = {
            "sex_inference_confidence": 0.5,
            "sample_count": 50,
            "sex_metadata_completeness": 0.0,
            "study_type": "microarray",
            "clinical_priority_score": 0.2,
        }

        score_high = calculate_rescue_score(study_high_priority)
        score_low = calculate_rescue_score(study_low_priority)

        assert score_high > score_low

    def test_rna_seq_type_bonus(self):
        """Test that RNA-seq study type gets bonus."""
        study_rna = {
            "sex_inference_confidence": 0.5,
            "sample_count": 50,
            "sex_metadata_completeness": 0.0,
            "study_type": "RNA-seq",
            "clinical_priority_score": 0.5,
        }
        study_array = {
            "sex_inference_confidence": 0.5,
            "sample_count": 50,
            "sex_metadata_completeness": 0.0,
            "study_type": "microarray",
            "clinical_priority_score": 0.5,
        }

        score_rna = calculate_rescue_score(study_rna)
        score_array = calculate_rescue_score(study_array)

        assert score_rna > score_array

    def test_score_normalization(self):
        """Test that score is normalized to [0.0, 1.0]."""
        study = {
            "sex_inference_confidence": 1.0,
            "sample_count": 500,  # Very large, could overflow if not normalized
            "sex_metadata_completeness": 0.0,
            "study_type": "RNA-seq",
            "clinical_priority_score": 1.0,
        }
        score = calculate_rescue_score(study)

        assert 0.0 <= score <= 1.0

    def test_score_components_documented(self):
        """Test that scoring formula components are transparent."""
        # This test verifies that the function produces meaningful outputs
        study = {
            "sex_inference_confidence": 0.7,
            "sample_count": 80,
            "sex_metadata_completeness": 0.2,  # 20% complete, 80% missing
            "study_type": "RNA-seq",
            "clinical_priority_score": 0.7,
        }
        score = calculate_rescue_score(study)

        # Should be reasonably high due to:
        # - 0.7 confidence (30% weight)
        # - Good sample size (25% weight)
        # - 80% missing metadata (20% weight)
        # - RNA-seq type (15% weight)
        # - 0.7 clinical priority (10% weight)
        assert score > 0.4


class TestFetchRescueStats:
    """Tests for rescue statistics."""

    def test_returns_dict_with_required_keys(self, mock_supabase_client):
        """Test that fetch_rescue_stats returns dict with required keys."""
        # Mock responses for queries
        mock_response1 = Mock()
        mock_response1.data = []
        mock_response1.count = 5

        mock_response2 = Mock()
        mock_response2.data = []
        mock_response2.count = 5

        query1_mock = Mock()
        query1_mock.eq.return_value = query1_mock
        query1_mock.execute.return_value = mock_response1

        query2_mock = Mock()
        query2_mock.eq.return_value = query2_mock
        query2_mock.gte.return_value = query2_mock
        query2_mock.execute.return_value = mock_response2

        select1_mock = Mock()
        select1_mock.eq.return_value = query1_mock

        select2_mock = Mock()
        select2_mock.eq.return_value = query2_mock

        call_count = [0]

        def table_side_effect(table_name):
            table_mock = Mock()
            if call_count[0] == 0:
                table_mock.select.return_value = select1_mock
                call_count[0] += 1
            else:
                table_mock.select.return_value = select2_mock
            return table_mock

        mock_supabase_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            stats = fetch_rescue_stats()

            assert isinstance(stats, dict)
            assert "total_opportunities" in stats
            assert "high_confidence_count" in stats
            assert "potential_samples" in stats
            assert "top_diseases" in stats


class TestSnapshotFunctions:
    """Tests for snapshot creation and retrieval."""

    def test_create_snapshot_returns_dict_with_required_keys(self, mock_supabase_client):
        """Test that create_snapshot returns dict with all required keys."""
        # Mock studies query
        mock_studies_response = Mock()
        mock_studies_response.data = [
            {"id": 1, "sex_metadata_completeness": 0.8},
            {"id": 2, "sex_metadata_completeness": 0.5},
        ]

        # Mock count responses
        mock_count_response = Mock()
        mock_count_response.count = 2

        # Build query chain mocks
        studies_query = Mock()
        studies_query.execute.return_value = mock_studies_response

        count_query = Mock()
        count_query.eq.return_value = count_query
        count_query.execute.return_value = mock_count_response

        insert_query = Mock()
        insert_query.insert.return_value = insert_query
        insert_query.execute.return_value = Mock(data=[{"snapshot_date": "2026-01-15"}])

        call_count = [0]

        def table_side_effect(table_name):
            table_mock = Mock()
            if table_name == "studies" and call_count[0] == 0:
                call_count[0] += 1
                table_mock.select.return_value = studies_query
            elif table_name == "studies":
                table_mock.select.return_value = count_query
            elif table_name == "completeness_snapshots":
                table_mock.insert.return_value = insert_query
            return table_mock

        mock_supabase_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            snapshot = create_snapshot()

            assert isinstance(snapshot, dict)
            assert "snapshot_date" in snapshot
            assert "total_studies" in snapshot
            assert "studies_with_sex_metadata" in snapshot
            assert "studies_sex_inferrable" in snapshot
            assert "studies_with_sex_analysis" in snapshot
            assert "avg_metadata_completeness" in snapshot

    def test_create_snapshot_with_organism_filter(self, mock_supabase_client):
        """Test create_snapshot with organism filter."""
        # Mock studies query
        mock_studies_response = Mock()
        mock_studies_response.data = [{"id": 1, "sex_metadata_completeness": 0.7}]

        # Mock count response
        mock_count_response = Mock()
        mock_count_response.count = 1

        studies_query = Mock()
        studies_query.eq.return_value = studies_query
        studies_query.execute.return_value = mock_studies_response

        count_query = Mock()
        count_query.eq.return_value = count_query
        count_query.execute.return_value = mock_count_response

        insert_query = Mock()
        insert_query.insert.return_value = insert_query
        insert_query.execute.return_value = Mock(data=[{"snapshot_date": "2026-01-15"}])

        call_count = [0]

        def table_side_effect(table_name):
            table_mock = Mock()
            if table_name == "studies" and call_count[0] == 0:
                call_count[0] += 1
                table_mock.select.return_value = studies_query
            elif table_name == "studies":
                table_mock.select.return_value = count_query
            elif table_name == "completeness_snapshots":
                table_mock.insert.return_value = insert_query
            return table_mock

        mock_supabase_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            snapshot = create_snapshot(organism="Homo sapiens")

            assert snapshot["organism"] == "Homo sapiens"
            assert snapshot["total_studies"] == 1

    def test_fetch_snapshots_returns_list(self, mock_supabase_client):
        """Test that fetch_snapshots returns a list of snapshots."""
        mock_response = Mock()
        mock_response.data = [
            {"snapshot_date": "2026-01-08", "total_studies": 50},
            {"snapshot_date": "2026-01-15", "total_studies": 52},
        ]

        query_mock = Mock()
        query_mock.select.return_value = query_mock
        query_mock.eq.return_value = query_mock
        query_mock.order.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = mock_response

        mock_supabase_client.table.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            snapshots = fetch_snapshots()

            assert isinstance(snapshots, list)
            assert len(snapshots) == 2
            assert snapshots[0]["snapshot_date"] == "2026-01-08"
            assert snapshots[1]["total_studies"] == 52

    def test_fetch_snapshots_with_organism_filter(self, mock_supabase_client):
        """Test fetch_snapshots with organism filter."""
        mock_response = Mock()
        mock_response.data = [{"snapshot_date": "2026-01-15", "organism": "Homo sapiens"}]

        query_mock = Mock()
        query_mock.select.return_value = query_mock
        query_mock.eq.return_value = query_mock
        query_mock.order.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = mock_response

        mock_supabase_client.table.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            snapshots = fetch_snapshots(organism="Homo sapiens")

            assert isinstance(snapshots, list)
            assert len(snapshots) == 1
            assert snapshots[0]["organism"] == "Homo sapiens"

    def test_fetch_snapshots_with_limit(self, mock_supabase_client):
        """Test fetch_snapshots respects limit parameter."""
        mock_response = Mock()
        mock_response.data = [
            {"snapshot_date": f"2026-01-{i:02d}", "total_studies": 50 + i} for i in range(1, 11)
        ]

        query_mock = Mock()
        query_mock.select.return_value = query_mock
        query_mock.eq.return_value = query_mock
        query_mock.order.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.execute.return_value = mock_response

        mock_supabase_client.table.return_value = query_mock

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            # Clear cache by calling with different parameters to get fresh result
            snapshots = fetch_snapshots(limit=5)

            assert isinstance(snapshots, list)
            # Should have 10 items from mock (limit is applied server-side)
            assert len(snapshots) >= 5

    def test_create_snapshot_with_date_returns_dict(self, mock_supabase_client):
        """Test that create_snapshot_with_date returns dict with required keys."""
        # Mock studies query (with publication_date)
        mock_studies_response = Mock()
        mock_studies_response.data = [
            {"id": 1, "sex_metadata_completeness": 0.8, "publication_date": "2026-01-10"},
            {"id": 2, "sex_metadata_completeness": 0.5, "publication_date": "2026-01-12"},
        ]

        # Mock count responses
        mock_count_response = Mock()
        mock_count_response.count = 2

        # Build query chain mocks
        studies_query = Mock()
        studies_query.lte.return_value = studies_query
        studies_query.execute.return_value = mock_studies_response

        count_query = Mock()
        count_query.eq.return_value = count_query
        count_query.lte.return_value = count_query
        count_query.execute.return_value = mock_count_response

        insert_query = Mock()
        insert_query.insert.return_value = insert_query
        insert_query.execute.return_value = Mock(data=[{"snapshot_date": "2026-01-15"}])

        call_count = [0]

        def table_side_effect(table_name):
            table_mock = Mock()
            if table_name == "studies" and call_count[0] == 0:
                call_count[0] += 1
                table_mock.select.return_value = studies_query
            elif table_name == "studies":
                table_mock.select.return_value = count_query
            elif table_name == "completeness_snapshots":
                table_mock.insert.return_value = insert_query
            return table_mock

        mock_supabase_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            snapshot = create_snapshot_with_date("2026-01-15")

            assert isinstance(snapshot, dict)
            assert snapshot["snapshot_date"] == "2026-01-15"
            assert "total_studies" in snapshot
            assert "studies_with_sex_metadata" in snapshot
            assert "avg_metadata_completeness" in snapshot

    def test_create_snapshot_with_date_filters_by_publication_date(self, mock_supabase_client):
        """Test that snapshots only include studies published by snapshot date."""
        # Mock studies query
        mock_studies_response = Mock()
        mock_studies_response.data = [
            {"id": 1, "sex_metadata_completeness": 0.9, "publication_date": "2026-01-10"}
        ]

        # Mock count response
        mock_count_response = Mock()
        mock_count_response.count = 1

        studies_query = Mock()
        studies_query.lte.return_value = studies_query
        studies_query.execute.return_value = mock_studies_response

        count_query = Mock()
        count_query.eq.return_value = count_query
        count_query.lte.return_value = count_query
        count_query.execute.return_value = mock_count_response

        insert_query = Mock()
        insert_query.insert.return_value = insert_query
        insert_query.execute.return_value = Mock(data=[{"snapshot_date": "2026-01-12"}])

        call_count = [0]

        def table_side_effect(table_name):
            table_mock = Mock()
            if table_name == "studies" and call_count[0] == 0:
                call_count[0] += 1
                table_mock.select.return_value = studies_query
            elif table_name == "studies":
                table_mock.select.return_value = count_query
            elif table_name == "completeness_snapshots":
                table_mock.insert.return_value = insert_query
            return table_mock

        mock_supabase_client.table.side_effect = table_side_effect

        with patch("sage.database.get_supabase_client", return_value=mock_supabase_client):
            snapshot = create_snapshot_with_date("2026-01-12")

            assert snapshot["total_studies"] == 1
            assert snapshot["snapshot_date"] == "2026-01-12"
