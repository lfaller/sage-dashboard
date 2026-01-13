"""Database module for Supabase integration."""
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    """
    Initialize and cache Supabase client.

    Credentials are read from Streamlit secrets:
    - connections.supabase.SUPABASE_URL
    - connections.supabase.SUPABASE_KEY

    Returns:
        Supabase client instance (cached)

    Raises:
        KeyError: If secrets are not configured
    """
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)


@st.cache_data(ttl=3600)
def fetch_overview_stats() -> dict:
    """
    Fetch overview statistics from database.

    Returns:
        Dict with keys:
        - total_studies: Total number of studies
        - with_sex_metadata: Studies with sex metadata completeness > 0
        - sex_inferrable: Studies where sex can be inferred
        - with_sex_analysis: Studies that report sex analysis

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    # Total studies
    total_response = client.table("studies").select("id", count="exact").execute()
    total = total_response.count or 0

    # With sex metadata
    with_sex_response = (
        client.table("studies")
        .select("id", count="exact")
        .gt("sex_metadata_completeness", 0)
        .execute()
    )
    with_sex = with_sex_response.count or 0

    # Sex inferrable
    inferrable_response = (
        client.table("studies").select("id", count="exact").eq("sex_inferrable", True).execute()
    )
    inferrable = inferrable_response.count or 0

    # With sex analysis
    analyzed_response = (
        client.table("studies")
        .select("id", count="exact")
        .eq("reports_sex_analysis", True)
        .execute()
    )
    analyzed = analyzed_response.count or 0

    return {
        "total_studies": total,
        "with_sex_metadata": with_sex,
        "sex_inferrable": inferrable,
        "with_sex_analysis": analyzed,
    }


@dataclass
class StudyFilter:
    """Filter criteria for study search."""

    organism: Optional[str] = None
    has_sex_metadata: Optional[bool] = None
    sex_inferrable: Optional[bool] = None
    limit: int = 100


def search_studies(
    organism: Optional[str] = None,
    has_sex_metadata: Optional[bool] = None,
    sex_inferrable: Optional[bool] = None,
    limit: int = 100,
) -> list:
    """
    Search studies with optional filters.

    Args:
        organism: Filter by organism (e.g., "Homo sapiens")
        has_sex_metadata: Filter by sex metadata presence
        sex_inferrable: Filter by sex inferrability
        limit: Maximum number of results (default 100)

    Returns:
        List of study records matching filters

    Raises:
        Exception: If database query fails
    """
    result = search_studies_advanced(
        organism=organism,
        has_sex_metadata=has_sex_metadata,
        sex_inferrable=sex_inferrable,
        limit=limit,
    )
    return result["results"]


def search_studies_advanced(
    search: Optional[str] = None,
    organism: Optional[str] = None,
    study_type: Optional[str] = None,
    has_sex_metadata: Optional[bool] = None,
    sex_inferrable: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    Advanced search with multiple filters and pagination.

    Args:
        search: Text search in title, summary, or geo_accession (case-insensitive)
        organism: Filter by organism
        study_type: Filter by study type (RNA-seq, microarray, etc.)
        has_sex_metadata: Filter by sex metadata presence
        sex_inferrable: Filter by sex inferrability
        limit: Maximum number of results (default 100)
        offset: Number of results to skip for pagination (default 0)

    Returns:
        Dict with:
        - results: List of matching studies
        - total: Total count of matching studies (before pagination)

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    query = client.table("studies").select(
        "id",
        "geo_accession",
        "title",
        "organism",
        "study_type",
        "sample_count",
        "has_sex_metadata",
        "sex_metadata_completeness",
        "sex_inferrable",
        "summary",
        "platform",
        "pubmed_id",
        count="exact",
    )

    # Text search: search across title, summary, and geo_accession
    if search is not None and search.strip():
        search_term = f"%{search}%"
        query = query.ilike("title", search_term)

    # Apply filters
    if organism is not None:
        query = query.eq("organism", organism)

    if study_type is not None:
        query = query.eq("study_type", study_type)

    if has_sex_metadata is not None:
        query = query.eq("has_sex_metadata", has_sex_metadata)

    if sex_inferrable is not None:
        query = query.eq("sex_inferrable", sex_inferrable)

    # Get total count before pagination
    response = query.limit(limit).offset(offset).execute()

    return {
        "results": response.data or [],
        "total": response.count or 0,
    }


@st.cache_data(ttl=3600)
def get_filter_options() -> dict:
    """
    Get unique values for filter dropdowns.

    Returns:
        Dict with:
        - organisms: List of unique organism names
        - study_types: List of unique study types
        - platforms: List of unique platforms

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    # Get unique organisms
    organisms_response = client.table("studies").select("organism", count="exact").execute()
    organisms = list(
        {item["organism"] for item in (organisms_response.data or []) if item.get("organism")}
    )

    # Get unique study types
    study_types_response = client.table("studies").select("study_type").execute()
    study_types = list(
        {item["study_type"] for item in (study_types_response.data or []) if item.get("study_type")}
    )

    # Get unique platforms
    platforms_response = client.table("studies").select("platform").execute()
    platforms = list(
        {item["platform"] for item in (platforms_response.data or []) if item.get("platform")}
    )

    return {
        "organisms": sorted(organisms),
        "study_types": sorted(study_types),
        "platforms": sorted(platforms),
    }


def get_study_by_accession(accession: str) -> Optional[dict]:
    """
    Get a single study by its GEO accession.

    Args:
        accession: GEO accession (e.g., "GSE123001")

    Returns:
        Study record if found, None otherwise

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    response = client.table("studies").select("*").eq("geo_accession", accession).execute()

    studies = response.data or []
    return studies[0] if studies else None
