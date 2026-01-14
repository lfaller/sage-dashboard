#!/usr/bin/env python3
"""Update sex inferrability for all studies in database.

This script analyzes all studies in the database using the sex inference module
and updates their sex_inferrable and sex_inference_confidence fields.

Usage:
    # Test on 10 studies without updating database
    python scripts/update_sex_inferrability.py --dry-run --limit 10

    # Update all studies
    python scripts/update_sex_inferrability.py

    # Incremental processing (update first 100)
    python scripts/update_sex_inferrability.py --limit 100
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.database import get_supabase_client  # noqa: E402
from sage.sex_inference import infer_from_metadata  # noqa: E402


def fetch_all_studies(limit: Optional[int] = None) -> List[Dict]:
    """Fetch all studies from database.

    Args:
        limit: Optional limit on number of studies to fetch

    Returns:
        List of study records

    Raises:
        Exception: If database query fails
    """
    client = get_supabase_client()

    query = client.table("studies").select(
        "id, geo_accession, title, organism, study_type, sample_count, "
        "has_sex_metadata, sex_metadata_completeness"
    )

    if limit:
        query = query.limit(limit)

    response = query.execute()
    return response.data or []


def update_study_inference(study_id: int, inference_result: Dict) -> bool:
    """Update a single study with inference results.

    Args:
        study_id: ID of study to update
        inference_result: Dict with sex_inferrable and sex_inference_confidence

    Returns:
        True if update succeeded, False otherwise
    """
    client = get_supabase_client()

    try:
        update_data = {
            "sex_inferrable": inference_result["sex_inferrable"],
            "sex_inference_confidence": inference_result["sex_inference_confidence"],
        }

        response = client.table("studies").update(update_data).eq("id", study_id).execute()

        return response.data is not None

    except Exception as e:
        print(f"  Error updating study {study_id}: {e}", file=sys.stderr)
        return False


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Update sex inferrability for all studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run analysis without updating database"
    )
    parser.add_argument("--limit", type=int, help="Limit number of studies (for testing)")

    args = parser.parse_args()

    print("=" * 70)
    print("Sex Inferrability Update Script")
    print("=" * 70)

    if args.dry_run:
        print("[DRY RUN MODE - No database updates will be performed]")
    print()

    # Fetch studies
    print("Fetching studies from database...")
    studies = fetch_all_studies(limit=args.limit)
    print(f"Found {len(studies)} studies to analyze")

    if not studies:
        print("No studies found. Exiting.")
        return 0

    # Process each study
    print("\nAnalyzing studies...")
    print("-" * 70)

    stats = {
        "total": len(studies),
        "inferrable": 0,
        "high_confidence": 0,  # >= 0.7
        "medium_confidence": 0,  # 0.5-0.7
        "low_confidence": 0,  # < 0.5
        "updated": 0,
        "errors": 0,
    }

    for i, study in enumerate(studies, 1):
        # Print progress every 10 studies
        if i % 10 == 0:
            progress = i / len(studies) * 100
            print(f"  Progress: {i}/{len(studies)} ({progress:.1f}%)")

        try:
            inference_result = infer_from_metadata(study)

            if inference_result["sex_inferrable"]:
                stats["inferrable"] += 1

                confidence = inference_result["sex_inference_confidence"]
                if confidence >= 0.7:
                    stats["high_confidence"] += 1
                elif confidence >= 0.5:
                    stats["medium_confidence"] += 1
                else:
                    stats["low_confidence"] += 1

            if not args.dry_run:
                success = update_study_inference(study["id"], inference_result)
                if success:
                    stats["updated"] += 1
                else:
                    stats["errors"] += 1

        except Exception as e:
            accession = study.get("geo_accession", f"ID:{study.get('id')}")
            print(f"  Error analyzing {accession}: {e}", file=sys.stderr)
            stats["errors"] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total studies analyzed:        {stats['total']}")
    print(
        f"Inferrable studies:            {stats['inferrable']} ({stats['inferrable']/stats['total']*100:.1f}%)"
    )
    if stats["inferrable"] > 0:
        print(f"  - High confidence (≥0.7):     {stats['high_confidence']}")
        print(f"  - Medium confidence (0.5-0.7):{stats['medium_confidence']}")
        print(f"  - Low confidence (<0.5):       {stats['low_confidence']}")

    if not args.dry_run:
        print(f"\nDatabase updates successful:   {stats['updated']}")
        print(f"Errors during update:          {stats['errors']}")
    else:
        print("\n[DRY RUN - No database updates were performed]")

    print("=" * 70)

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
