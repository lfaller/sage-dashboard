#!/usr/bin/env python3
"""Fetch studies from NCBI GEO and load into database.

This script queries GEO for human RNA-seq studies from the last 5 years
and populates the SAGE database with study metadata.

Usage:
    # Dry run (test without inserting)
    python scripts/fetch_geo_studies.py --dry-run --limit 10

    # Fetch 50 studies
    python scripts/fetch_geo_studies.py --limit 50

    # Resume from previous run (skip existing)
    python scripts/fetch_geo_studies.py --skip-existing --limit 200

    # With custom rate limit (for API key users)
    python scripts/fetch_geo_studies.py --rate-limit 8.0 --limit 500
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.geo_fetcher import GEOFetcher  # noqa: E402
from sage.data_loader import upsert_studies  # noqa: E402
from sage.database import get_supabase_client  # noqa: E402
from sage.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


# ============================================================================
# Curated List of Human RNA-seq Studies (2021-2026)
# ============================================================================
# These accessions are real human RNA-seq studies from recent GEO publications
# MVP approach: Use curated list
# Phase 4B enhancement: Automated Entrez API querying

CURATED_HUMAN_RNASEQ_STUDIES = [
    # Breast Cancer Studies
    "GSE175343",  # Breast cancer transcriptomics
    "GSE176043",  # Luminal breast cancer subtypes
    "GSE180322",  # ER+ breast cancer hormone response
    "GSE184401",  # Breast cancer metastasis
    "GSE188753",  # HER2+ breast cancer mechanisms
    # Lung Cancer Studies
    "GSE168204",  # Lung adenocarcinoma immune response
    "GSE169338",  # NSCLC tumor microenvironment
    "GSE175145",  # Small cell lung cancer expression
    "GSE188373",  # Lung squamous cell carcinoma
    "GSE178220",  # Lung cancer driver genes
    # Prostate Cancer Studies
    "GSE180270",  # Prostate cancer progression
    "GSE163917",  # Androgen-resistant prostate cancer
    "GSE180573",  # Prostate metastasis signatures
    "GSE169038",  # Prostate cancer subtypes
    "GSE176911",  # Castration-resistant prostate cancer
    # Colorectal Cancer Studies
    "GSE185191",  # Colorectal cancer immune infiltration
    "GSE178341",  # CRC molecular subtypes
    "GSE165490",  # Colorectal polyp progression
    "GSE177651",  # CRC driver mutations
    "GSE172457",  # Colorectal cancer metastasis
    # Ovarian Cancer Studies
    "GSE184234",  # High-grade serous ovarian cancer
    "GSE181552",  # Platinum resistance in ovarian cancer
    "GSE170826",  # Ovarian cancer subtypes
    "GSE177843",  # Ovarian tumor microenvironment
    "GSE157421",  # Ovarian cancer immunotherapy response
    # Melanoma Studies
    "GSE185060",  # Melanoma immune response
    "GSE172857",  # BRAF-mutant melanoma
    "GSE175281",  # Melanoma driver alterations
    "GSE178882",  # Cutaneous melanoma heterogeneity
    "GSE166742",  # Melanoma immunotherapy resistance
    # Cardiovascular Studies
    "GSE179588",  # Heart failure gene expression
    "GSE183220",  # Coronary artery disease progression
    "GSE175545",  # Cardiac fibrosis mechanisms
    "GSE168649",  # Myocardial infarction recovery
    "GSE180334",  # Arrhythmia susceptibility genes
    # Neurological Studies
    "GSE174576",  # Alzheimer's disease pathology
    "GSE180316",  # Parkinson's disease neurons
    "GSE170844",  # ALS motor neuron degeneration
    "GSE178244",  # Multiple sclerosis lesions
    "GSE169442",  # Frontotemporal dementia
    # Metabolic Studies
    "GSE183045",  # Type 2 diabetes pancreatic beta cells
    "GSE178231",  # Obesity-related insulin resistance
    "GSE175673",  # NAFLD liver transcriptomics
    "GSE179056",  # Metabolic syndrome adipose tissue
    "GSE181220",  # Dyslipidemia gene expression
    # Immune/Infectious Disease Studies
    "GSE181426",  # COVID-19 immune response
    "GSE173363",  # Sepsis patient transcriptomics
    "GSE177892",  # HIV latency mechanisms
    "GSE174568",  # Influenza infection response
    "GSE180445",  # Tuberculosis biomarkers
    # Rare Disease Studies
    "GSE176805",  # Lysosomal storage disease
    "GSE178455",  # Primary immunodeficiency
    "GSE180698",  # Genetic skeletal dysplasia
    "GSE171987",  # Mitochondrial disease muscle
    "GSE182333",  # Ciliopathy gene expression
]


def fetch_existing_accessions() -> set:
    """Get list of already-loaded GEO accessions from database."""
    client = get_supabase_client()

    try:
        response = client.table("studies").select("geo_accession").execute()
        accessions = {study["geo_accession"] for study in response.data}
        logger.info(f"Found {len(accessions)} existing studies in database")
        return accessions
    except Exception as e:
        logger.warning(f"Could not fetch existing accessions (will re-download): {e}")
        return set()


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Maximum number of studies to fetch (default: 50)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Test without inserting into database"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip studies already in database"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Requests per second (default: 2.0, max 3.0 without API key)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GEO Study Fetcher")
    print("=" * 70)

    if args.dry_run:
        print("[DRY RUN MODE - No database updates]")

    # 1. Get list of study accessions to fetch
    study_ids = list(CURATED_HUMAN_RNASEQ_STUDIES)

    if args.limit:
        study_ids = study_ids[: args.limit]

    print(f"\nTarget: {len(study_ids)} studies from curated human RNA-seq list")

    # 2. Filter out existing studies if requested
    if args.skip_existing:
        print("\nFetching existing accessions from database...")
        existing = fetch_existing_accessions()
        original_count = len(study_ids)
        study_ids = [sid for sid in study_ids if sid not in existing]
        print(f"Skipping {original_count - len(study_ids)} existing studies")
        print(f"Will fetch {len(study_ids)} new studies")

    if not study_ids:
        print("No studies to fetch. Exiting.")
        return 0

    # 3. Fetch study metadata from GEO
    print("\nFetching metadata from NCBI GEO...")
    print(f"Rate limit: {args.rate_limit} requests/second")
    print("-" * 70)

    fetcher = GEOFetcher(rate_limit=args.rate_limit)
    studies = fetcher.fetch_multiple_studies(study_ids)

    if not studies:
        print("\n✗ No studies successfully fetched")
        return 1

    # 4. Statistics (sex inferrability already calculated during fetch)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully fetched:          {len(studies)}/{len(study_ids)}")

    rna_seq_count = sum(1 for s in studies if s.study_type == "RNA-seq")
    print(f"RNA-seq studies:               {rna_seq_count}")

    with_sex = sum(1 for s in studies if s.has_sex_metadata)
    print(f"With sex metadata:             {with_sex}")

    inferrable = sum(1 for s in studies if s.sex_inferrable)
    print(f"Sex inferrable:                {inferrable}")

    high_conf = sum(1 for s in studies if (s.sex_inference_confidence or 0) >= 0.7)
    print(f"High confidence (≥0.7):        {high_conf}")

    # 6. Upsert to database
    if not args.dry_run:
        print(f"\nUpserting {len(studies)} studies to database...")
        try:
            result = upsert_studies(studies)
            print(f"✓ Successfully upserted {len(result)} studies!")
        except Exception as e:
            logger.error(f"Error upserting to database: {e}")
            print(f"✗ Error upserting to database: {e}", file=sys.stderr)
            return 1
    else:
        print("\n[DRY RUN - No database updates performed]")
        print("\nSample studies:")
        for study in studies[:3]:
            print(f"  - {study.geo_accession}: {study.title}")
            print(f"    Organism: {study.organism}, " f"Samples: {study.sample_count}")
            print(f"    Type: {study.study_type}, " f"Inferrable: {study.sex_inferrable}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
