"""GEO data fetcher using GEOparse library.

This module fetches study metadata from NCBI GEO database using the GEOparse
library, with built-in rate limiting to respect NCBI API limits.

NCBI E-utilities limits:
- Without API key: 3 requests/second
- With API key: 10 requests/second

Default rate limit is conservative 2 requests/second to be safe.
"""

import time
from typing import Optional, List, Tuple

import GEOparse

from sage.data_loader import Study
from sage.sex_inference import (
    analyze_sample_names,
    analyze_sample_characteristics,
    merge_sex_analyses,
    calculate_confidence,
)
from sage.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Simple rate limiter for NCBI API compliance.

    Ensures minimum delay between requests to respect NCBI API limits.
    """

    def __init__(self, requests_per_second: float = 2.0):
        """Initialize rate limiter.

        Args:
            requests_per_second: Target requests per second (default: 2.0)
        """
        self.delay = 1.0 / requests_per_second
        self.last_request = 0.0

    def wait(self):
        """Ensure minimum delay since last request."""
        now = time.time()
        elapsed = now - self.last_request

        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        self.last_request = time.time()


class GEOFetcher:
    """Fetches study metadata from NCBI GEO database using GEOparse."""

    # Allowed organisms for SAGE (sex metadata research focus)
    ALLOWED_ORGANISMS = {"Homo sapiens", "Mus musculus"}
    # Allowed study types for SAGE (RNA-seq is primary focus)
    ALLOWED_STUDY_TYPES = {"RNA-seq"}

    def __init__(self, rate_limit: float = 2.0, use_cache: bool = True):
        """Initialize GEO fetcher.

        Args:
            rate_limit: Requests per second (default: 2.0)
            use_cache: Whether to use GEOparse local cache (default: True)
        """
        self.rate_limiter = RateLimiter(rate_limit)
        self.use_cache = use_cache
        logger.info(f"GEOFetcher initialized (rate: {rate_limit} req/sec)")

    def fetch_study(self, geo_accession: str, retry_count: int = 3) -> Optional[Study]:
        """Fetch single study by GEO accession.

        Args:
            geo_accession: GEO series accession (e.g., "GSE123456")
            retry_count: Number of retries on failure (default: 3)

        Returns:
            Study object or None if fetch failed or organism not allowed
        """
        logger.debug(f"Fetching {geo_accession}")
        self.rate_limiter.wait()

        for attempt in range(retry_count):
            try:
                # Fetch using GEOparse
                gse = GEOparse.get_GEO(geo_accession, destdir="./local/geo_cache", silent=True)

                # Convert to Study object (returns None if organism not allowed)
                study = self._gse_to_study(gse)
                if study is None:
                    return None

                logger.info(f"✓ Fetched {geo_accession}: {study.title[:50]}...")
                return study

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt+1}/{retry_count} failed for " f"{geo_accession}: {e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"✗ Failed to fetch {geo_accession} after " f"{retry_count} attempts"
                    )
                    return None

    def fetch_multiple_studies(
        self, accessions: List[str], skip_existing: Optional[set] = None
    ) -> List[Study]:
        """Fetch multiple studies with progress tracking.

        Args:
            accessions: List of GEO accessions to fetch
            skip_existing: Optional set of accessions to skip

        Returns:
            List of successfully fetched Study objects
        """
        if skip_existing:
            accessions = [acc for acc in accessions if acc not in skip_existing]

        studies = []
        total = len(accessions)

        for i, accession in enumerate(accessions, 1):
            logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
            study = self.fetch_study(accession)

            if study:
                studies.append(study)

        logger.info(f"Successfully fetched {len(studies)}/{total} studies")
        return studies

    def _gse_to_study(self, gse) -> Optional[Study]:
        """Convert GEOparse GSE object to Study dataclass.

        Args:
            gse: GEOparse GSE object

        Returns:
            Study dataclass instance, or None if organism is not allowed
        """
        # Extract basic metadata
        geo_accession = gse.name
        title = gse.metadata.get("title", ["Unknown"])[0]

        # Organism: try multiple metadata fields
        organism = gse.metadata.get("organism", gse.metadata.get("organism_ch1", ["Unknown"]))[0]
        if organism == "Unknown":
            # Try taxid fields (9606 = human, 10090 = mouse, etc.)
            taxid_list = gse.metadata.get("sample_taxid", gse.metadata.get("platform_taxid", [""]))
            taxid = taxid_list[0] if taxid_list else ""
            if taxid == "9606":
                organism = "Homo sapiens"
            elif taxid == "10090":
                organism = "Mus musculus"
            elif taxid:
                organism = f"taxid:{taxid}"

        # Filter: Only allow human and mouse studies
        if organism not in self.ALLOWED_ORGANISMS:
            logger.info(f"Skipping {geo_accession}: organism '{organism}' not in allowed list")
            return None

        sample_count = len(gse.gsms)

        # Optional fields
        summary = gse.metadata.get("summary", [None])[0]
        platform = gse.metadata.get("platform_id", [None])[0]
        publication_date = gse.metadata.get("submission_date", [None])[0]
        pubmed_id = gse.metadata.get("pubmed_id", [None])[0]

        # Detect study type
        study_type = self._detect_study_type(gse)

        # Filter: Only allow RNA-seq studies (sex metadata focus)
        if study_type not in self.ALLOWED_STUDY_TYPES:
            logger.info(f"Skipping {geo_accession}: study type '{study_type}' not in allowed list")
            return None

        # Detect sex metadata (study-level only)
        has_sex_metadata, sex_completeness = detect_sex_metadata_from_gse(gse)

        # Calculate sex inferrability using full study context
        sex_inferrable, inference_confidence = self._calculate_sex_inferrability(
            gse, organism, sample_count, study_type, has_sex_metadata
        )

        return Study(
            geo_accession=geo_accession,
            title=title,
            organism=organism,
            sample_count=sample_count,
            summary=summary,
            platform=platform,
            study_type=study_type,
            publication_date=publication_date,
            pubmed_id=pubmed_id,
            has_sex_metadata=has_sex_metadata,
            sex_metadata_completeness=sex_completeness,
            sex_inferrable=sex_inferrable,
            sex_inference_confidence=inference_confidence,
        )

    def _detect_study_type(self, gse) -> Optional[str]:
        """Detect study type from GEO metadata.

        Args:
            gse: GEOparse GSE object

        Returns:
            "RNA-seq", "microarray", or None
        """
        study_types = gse.metadata.get("type", [])
        if not study_types:
            return None

        type_str = study_types[0].lower()
        if "sequencing" in type_str:
            return "RNA-seq"
        elif "array" in type_str:
            return "microarray"
        else:
            return None

    def _calculate_sex_inferrability(
        self,
        gse,
        organism: str,
        sample_count: int,
        study_type: Optional[str],
        has_sex_metadata: bool,
    ) -> Tuple[bool, float]:
        """Calculate sex inferrability based on study characteristics.

        Combines sample name analysis and characteristics with study metadata
        to determine if sex can be inferred from expression data (future phase).

        Args:
            gse: GEOparse GSE object
            organism: Organism name
            sample_count: Number of samples
            study_type: Study type (RNA-seq, microarray, etc.)
            has_sex_metadata: Whether sex metadata detected (names or characteristics)

        Returns:
            Tuple of (sex_inferrable: bool, confidence: float 0-1)
        """
        # Extract sample names and characteristics for analysis
        sample_names = []
        samples_characteristics = []
        for gsm_id in gse.gsms.keys():
            try:
                title = gse.gsms[gsm_id].metadata.get("title", [""])[0]
                chars = gse.gsms[gsm_id].metadata.get("characteristics_ch1", [])
                sample_names.append(title)
                samples_characteristics.append(chars)
            except Exception:
                continue

        # Analyze both sources and merge results
        names_analysis = analyze_sample_names(sample_names)
        chars_analysis = analyze_sample_characteristics(samples_characteristics)
        merged_analysis = merge_sex_analyses(chars_analysis, names_analysis)

        # Build factors dict for confidence calculation
        factors = {
            "is_rna_seq": study_type == "RNA-seq",
            "sample_count": sample_count,
            "has_sufficient_samples": sample_count >= 20,
            "is_human": organism == "Homo sapiens",
            "sample_name_confidence": merged_analysis["confidence"],
            "sample_name_pattern": merged_analysis["pattern"],
            "has_sex_metadata": has_sex_metadata,
        }

        # Calculate confidence using established formula
        confidence = calculate_confidence(factors)
        inferrable = confidence >= 0.5

        return inferrable, confidence


def detect_sex_metadata_from_gse(gse) -> Tuple[bool, float]:
    """Detect sex metadata presence from GSE object (study-level only).

    Analyzes both sample titles/names and characteristics_ch1 for sex indicators.
    Prioritizes characteristics as more explicit/reliable source.

    Args:
        gse: GEOparse GSE object

    Returns:
        Tuple of (has_sex_metadata: bool, completeness: float 0-1)
    """
    # Extract sample titles and characteristics from GSE metadata
    sample_names = []
    samples_characteristics = []
    for gsm_id in gse.gsms.keys():
        try:
            title = gse.gsms[gsm_id].metadata.get("title", [""])[0]
            chars = gse.gsms[gsm_id].metadata.get("characteristics_ch1", [])
            sample_names.append(title)
            samples_characteristics.append(chars)
        except Exception:
            continue

    if not sample_names:
        return False, 0.0

    # Analyze both sources and merge results
    names_analysis = analyze_sample_names(sample_names)
    chars_analysis = analyze_sample_characteristics(samples_characteristics)
    merged_analysis = merge_sex_analyses(chars_analysis, names_analysis)

    # Consider "clear" or "partial" patterns as having metadata
    has_metadata = merged_analysis["pattern"] in ["clear", "partial"]
    completeness = merged_analysis["confidence"]

    return has_metadata, completeness
