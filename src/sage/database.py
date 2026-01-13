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
    client = get_supabase_client()

    query = client.table("studies").select("*")

    if organism is not None:
        query = query.eq("organism", organism)

    if has_sex_metadata is not None:
        query = query.eq("has_sex_metadata", has_sex_metadata)

    if sex_inferrable is not None:
        query = query.eq("sex_inferrable", sex_inferrable)

    query = query.limit(limit)

    response = query.execute()
    return response.data or []
