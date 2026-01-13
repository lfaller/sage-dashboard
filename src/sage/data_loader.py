"""Data loading module for GEO studies."""
from dataclasses import asdict, dataclass
from typing import Optional

from sage.database import get_supabase_client


@dataclass
class Study:
    """Represents a genomics study from GEO."""

    geo_accession: str
    title: str
    organism: str
    sample_count: int
    summary: Optional[str] = None
    platform: Optional[str] = None
    study_type: Optional[str] = None
    publication_date: Optional[str] = None
    pubmed_id: Optional[str] = None
    has_sex_metadata: bool = False
    sex_metadata_completeness: float = 0.0
    sex_inferrable: bool = False
    sex_inference_confidence: Optional[float] = None
    reports_sex_analysis: bool = False
    reports_sex_difference: Optional[bool] = None


def calculate_sex_metadata_completeness(samples_with_sex: int, total_samples: int) -> float:
    """
    Calculate the fraction of samples with sex metadata.

    Args:
        samples_with_sex: Number of samples with sex information
        total_samples: Total number of samples

    Returns:
        Completeness as float between 0.0 and 1.0

    Raises:
        ValueError: If samples_with_sex > total_samples
    """
    if samples_with_sex > total_samples:
        raise ValueError("samples_with_sex cannot exceed total_samples")

    if total_samples == 0:
        return 0.0

    return samples_with_sex / total_samples


def parse_geo_metadata(geo_data: dict) -> Study:
    """
    Parse GEO API response into Study object.

    Args:
        geo_data: Raw GEO API response dictionary

    Returns:
        Study object with parsed metadata

    Raises:
        KeyError: If required fields are missing
    """
    accession = geo_data["accession"]
    title = geo_data["title"]
    gse = geo_data["gse"]

    organism = gse["organism"]
    sample_ids = gse.get("sample_id", [])
    sample_count = len(sample_ids)

    # Determine study type from type field
    study_types = gse.get("type", [])
    study_type = None
    if study_types:
        type_str = study_types[0].lower()
        if "sequencing" in type_str:
            study_type = "RNA-seq"
        elif "array" in type_str:
            study_type = "microarray"

    # Optional fields
    summary = geo_data.get("summary")
    platform = gse.get("platform")
    overall_design = gse.get("overall_design")

    return Study(
        geo_accession=accession,
        title=title,
        organism=organism,
        sample_count=sample_count,
        summary=summary or overall_design,
        platform=platform,
        study_type=study_type,
    )


def upsert_studies(studies: list[Study]) -> list[dict]:
    """
    Upsert studies into Supabase database.

    Performs an "upsert" - updates if exists (by geo_accession), inserts if new.

    Args:
        studies: List of Study objects to upsert

    Returns:
        List of upserted records with database IDs

    Raises:
        Exception: If database operation fails
    """
    if not studies:
        return []

    client = get_supabase_client()

    # Convert Study objects to dicts
    studies_data = [asdict(study) for study in studies]

    # Upsert with geo_accession as unique key
    response = client.table("studies").upsert(studies_data, on_conflict="geo_accession").execute()

    return response.data or []


def estimate_sex_inferrability(study: Study) -> bool:
    """
    Estimate if sex can be inferred from a study.

    Currently based on: RNA-seq data with sufficient sample size.

    Args:
        study: Study object to evaluate

    Returns:
        True if sex is likely inferrable, False otherwise
    """
    # RNA-seq studies with decent sample size are good candidates
    is_rna_seq = study.study_type == "RNA-seq"
    has_samples = study.sample_count >= 20

    return is_rna_seq and has_samples
