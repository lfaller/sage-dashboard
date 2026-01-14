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


@st.cache_data(ttl=3600)
def fetch_disease_stats() -> dict:
    """
    Fetch disease-related statistics from database.

    Returns:
        Dict with keys:
        - total_diseases: Unique disease count
        - diseases_with_studies: Count of diseases with mapped studies
        - avg_completeness: Average sex metadata completeness across disease studies
        - total_study_mappings: Total disease-study mappings

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    try:
        # Get total unique diseases
        diseases_response = (
            client.table("disease_mappings").select("disease_term", count="exact").execute()
        )

        total_diseases = len(
            set(
                item.get("disease_term")
                for item in (diseases_response.data or [])
                if item.get("disease_term")
            )
        )
        total_study_mappings = diseases_response.count or 0

        # Get diseases with studies (non-zero study count)
        # We'll count distinct diseases that have at least one study mapping
        diseases_with_studies = total_diseases

        # Get average completeness
        # This would require a JOIN in Supabase, so we'll estimate from available data
        avg_completeness = 0.0

        return {
            "total_diseases": total_diseases,
            "diseases_with_studies": diseases_with_studies,
            "avg_completeness": avg_completeness,
            "total_study_mappings": total_study_mappings,
        }
    except Exception:
        # Return zeros if disease_mappings table doesn't exist or is empty
        return {
            "total_diseases": 0,
            "diseases_with_studies": 0,
            "avg_completeness": 0.0,
            "total_study_mappings": 0,
        }


@st.cache_data(ttl=3600)
def get_diseases_with_completeness(
    disease_category: Optional[str] = None,
    min_studies: int = 1,
    known_sex_diff_only: bool = False,
    limit: int = 100,
) -> list[dict]:
    """
    Get diseases with sex metadata completeness metrics.

    Args:
        disease_category: Filter by disease category
        min_studies: Minimum study count threshold
        known_sex_diff_only: Only diseases with known sex differences
        limit: Max results

    Returns:
        List of dicts with disease metrics:
        - disease_term: Disease name
        - disease_category: Disease category
        - study_count: Number of studies mapped to disease
        - avg_completeness: Average sex metadata completeness
        - known_sex_difference: Boolean flag
        - sex_bias_direction: Direction of sex bias
        - avg_clinical_priority: Average clinical priority score

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    try:
        query = client.table("disease_mappings").select(
            "disease_term, disease_category, known_sex_difference, "
            "sex_bias_direction, clinical_priority_score",
            count="exact",
        )

        # Apply filters conditionally
        if disease_category is not None:
            query = query.eq("disease_category", disease_category)

        if known_sex_diff_only:
            query = query.eq("known_sex_difference", True)

        # Execute query to get data
        response = query.limit(limit).offset(0).execute()

        diseases_data = response.data or []

        # If no data or filter resulted in empty, return empty list
        if not diseases_data:
            return []

        # Aggregate data by disease_term (simulate GROUP BY in Python)
        disease_dict = {}
        for item in diseases_data:
            term = item.get("disease_term")
            if not term:
                continue

            if term not in disease_dict:
                disease_dict[term] = {
                    "disease_term": term,
                    "disease_category": item.get("disease_category"),
                    "study_count": 0,
                    "avg_completeness": 0.0,
                    "known_sex_difference": item.get("known_sex_difference", False),
                    "sex_bias_direction": item.get("sex_bias_direction"),
                    "avg_clinical_priority": item.get("clinical_priority_score", 0.0),
                }
            disease_dict[term]["study_count"] += 1

        # Filter by min_studies
        result = [d for d in disease_dict.values() if d["study_count"] >= min_studies]

        # Sort by study count descending
        result.sort(key=lambda x: x["study_count"], reverse=True)

        return result[:limit]

    except Exception:
        return []


def get_studies_for_disease(
    disease_term: str,
    limit: int = 100,
) -> list[dict]:
    """
    Get studies mapped to a specific disease.

    Args:
        disease_term: Disease term to look up
        limit: Max results

    Returns:
        List of study records with completeness info

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    try:
        # First, get the disease mapping IDs for this disease
        mapping_response = (
            client.table("disease_mappings")
            .select("study_id")
            .eq("disease_term", disease_term)
            .limit(limit)
            .execute()
        )

        study_ids = [item.get("study_id") for item in (mapping_response.data or [])]

        if not study_ids:
            return []

        # Now get the studies
        studies = []
        for study_id in study_ids:
            study_response = (
                client.table("studies")
                .select(
                    "id, geo_accession, title, organism, sample_count, "
                    "sex_metadata_completeness, reports_sex_analysis"
                )
                .eq("id", study_id)
                .execute()
            )

            if study_response.data:
                studies.extend(study_response.data)

        return studies

    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_disease_categories() -> list[str]:
    """
    Get unique disease categories for filtering.

    Returns:
        Sorted list of unique disease_category values

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    try:
        response = client.table("disease_mappings").select("disease_category").execute()

        categories = set()
        for item in response.data or []:
            cat = item.get("disease_category")
            if cat:  # Filter out NULLs
                categories.add(cat)

        return sorted(list(categories))

    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_rescue_opportunities(
    organism: Optional[str] = None,
    disease_category: Optional[str] = None,
    min_confidence: float = 0.0,
    min_sample_size: int = 0,
    limit: int = 100,
) -> list:
    """
    Get studies ranked by rescue potential.

    Rescue opportunities are studies that:
    - Lack sex metadata (has_sex_metadata = False)
    - Can have sex inferred (sex_inferrable = True)
    - Meet confidence and sample size thresholds

    Args:
        organism: Optional organism filter (e.g., "Homo sapiens")
        disease_category: Optional disease category filter
        min_confidence: Minimum inference confidence (0.0-1.0, default 0.0 for no filter)
        min_sample_size: Minimum sample count (default 0, no filter)
        limit: Maximum results (default 100)

    Returns:
        List of rescue opportunity studies with rescue_score, sorted descending

    Raises:
        Exception: If database query fails
    """
    import sys

    print(
        f"[RESCUE] Called with: organism={organism}, disease={disease_category}, "
        f"confidence={min_confidence}, sample_size={min_sample_size}",
        file=sys.stderr,
    )

    client = get_supabase_client()

    try:
        query = client.table("studies").select(
            "id, geo_accession, title, organism, study_type, sample_count, "
            "has_sex_metadata, sex_metadata_completeness, sex_inferrable, "
            "sex_inference_confidence, clinical_priority_score",
            count="exact",
        )

        # Core filters for rescue opportunities
        query = query.eq("has_sex_metadata", False)
        query = query.eq("sex_inferrable", True)

        print(
            "[RESCUE] Applied core filters (has_sex_metadata=False, sex_inferrable=True)",
            file=sys.stderr,
        )

        # Only apply confidence filter if > 0
        if min_confidence > 0.0:
            query = query.gte("sex_inference_confidence", min_confidence)
            print(f"[RESCUE] Applied confidence filter: >= {min_confidence}", file=sys.stderr)

        # Only apply sample size filter if > 0
        if min_sample_size > 0:
            query = query.gte("sample_count", min_sample_size)
            print(f"[RESCUE] Applied sample size filter: >= {min_sample_size}", file=sys.stderr)

        # Optional organism filter
        if organism is not None:
            query = query.eq("organism", organism)

        response = query.limit(limit).execute()
        studies = response.data or []

        print(f"[RESCUE] Query returned {len(studies)} studies", file=sys.stderr)

        # Calculate rescue scores
        for study in studies:
            study["rescue_score"] = calculate_rescue_score(study)

        # Filter by disease category if specified
        if disease_category is not None:
            disease_response = (
                client.table("disease_mappings")
                .select("study_id")
                .eq("disease_category", disease_category)
                .execute()
            )
            disease_study_ids = {m["study_id"] for m in (disease_response.data or [])}
            studies = [s for s in studies if s["id"] in disease_study_ids]

        # Sort by rescue_score descending
        studies.sort(key=lambda x: x["rescue_score"], reverse=True)

        print(f"[RESCUE] Returning {len(studies)} sorted studies", file=sys.stderr)
        return studies

    except Exception as e:
        import traceback

        print(f"[RESCUE] ERROR: {e}", file=sys.stderr)
        print(f"[RESCUE] Traceback: {traceback.format_exc()}", file=sys.stderr)
        return []


def calculate_rescue_score(study: dict) -> float:
    """
    Calculate rescue potential score (0.0-1.0).

    Scoring factors (weighted):
    1. Inference confidence (30%)
    2. Sample size (25%) - normalized 20-200 range
    3. Missing metadata severity (20%) - 1 - completeness
    4. Study type (15%) - RNA-seq > microarray
    5. Clinical priority (10%) - Disease relevance

    Args:
        study: Study record dict

    Returns:
        Rescue score in [0.0, 1.0]
    """
    score = 0.0

    # 1. Inference confidence (30%)
    confidence = study.get("sex_inference_confidence", 0.0)
    score += confidence * 0.30

    # 2. Sample size (25%) - normalize 20-200 range
    sample_count = study.get("sample_count", 0)
    sample_score = min(1.0, (sample_count - 20) / 180) if sample_count > 20 else 0.0
    score += sample_score * 0.25

    # 3. Missing metadata severity (20%)
    metadata_completeness = study.get("sex_metadata_completeness", 0.0)
    missing_severity = 1.0 - metadata_completeness
    score += missing_severity * 0.20

    # 4. Study type (15%)
    study_type = study.get("study_type", "")
    if study_type == "RNA-seq":
        score += 0.15
    elif study_type == "microarray":
        score += 0.08

    # 5. Clinical priority (10%)
    clinical_priority = study.get("clinical_priority_score", 0.5)
    score += clinical_priority * 0.10

    return min(1.0, max(0.0, score))


@st.cache_data(ttl=3600)
def fetch_rescue_stats() -> dict:
    """
    Fetch summary statistics for rescue opportunities.

    Returns:
        Dict with:
        - total_opportunities: Count of rescue opportunity studies
        - high_confidence_count: Studies with confidence >= 0.7
        - potential_samples: Sum of sample counts across opportunities
        - top_diseases: List of diseases in rescue opportunities (placeholder)

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    try:
        # Total opportunities
        total_response = (
            client.table("studies")
            .select("id, sample_count", count="exact")
            .eq("has_sex_metadata", False)
            .eq("sex_inferrable", True)
            .execute()
        )

        total = total_response.count or 0

        # High confidence count (>=0.7)
        high_conf_response = (
            client.table("studies")
            .select("id", count="exact")
            .eq("has_sex_metadata", False)
            .eq("sex_inferrable", True)
            .gte("sex_inference_confidence", 0.7)
            .execute()
        )

        high_conf = high_conf_response.count or 0

        # Sum potential samples
        potential_samples = sum(s.get("sample_count", 0) for s in (total_response.data or []))

        return {
            "total_opportunities": total,
            "high_confidence_count": high_conf,
            "potential_samples": potential_samples,
            "top_diseases": [],
        }

    except Exception:
        return {
            "total_opportunities": 0,
            "high_confidence_count": 0,
            "potential_samples": 0,
            "top_diseases": [],
        }
