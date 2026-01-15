#!/usr/bin/env python3
"""Backfill historical completeness snapshots from study submission dates.

This one-time script generates retroactive snapshots based on when studies
were submitted to GEO, allowing the Trends page to show historical data
without waiting for actual weekly snapshots.

For example, if a study was submitted on Jan 10, a snapshot created for Jan 10
will include that study (and all others submitted by that date), simulating
what the metrics would have been on that date.

Usage:
    # Backfill last 8 weeks (default)
    poetry run python scripts/backfill_snapshots.py

    # Backfill last 12 weeks
    poetry run python scripts/backfill_snapshots.py --weeks 12

    # Backfill last 4 weeks
    poetry run python scripts/backfill_snapshots.py --weeks 4

    # Dry run (test without inserting)
    poetry run python scripts/backfill_snapshots.py --dry-run
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.database import get_supabase_client, create_snapshot_with_date  # noqa: E402
from sage.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def get_study_date_range():
    """Get earliest and latest submission dates from studies in database.

    Returns:
        Tuple of (earliest_date, latest_date) in YYYY-MM-DD format,
        or (None, None) if no studies found.
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("studies")
            .select("publication_date")
            .order("publication_date", desc=False)
            .limit(1)
            .execute()
        )
        earliest = response.data[0]["publication_date"] if response.data else None

        response = (
            client.table("studies")
            .select("publication_date")
            .order("publication_date", desc=True)
            .limit(1)
            .execute()
        )
        latest = response.data[0]["publication_date"] if response.data else None

        return earliest, latest
    except Exception as e:
        logger.error(f"Error fetching study date range: {e}")
        return None, None


def generate_weekly_dates(end_date, weeks):
    """Generate weekly snapshot dates, working backwards from end_date.

    Args:
        end_date: End date in YYYY-MM-DD format
        weeks: Number of weeks to generate

    Returns:
        List of dates in YYYY-MM-DD format, ordered from oldest to newest
    """
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []

    for i in range(weeks):
        date = end - timedelta(weeks=i)
        dates.append(date.strftime("%Y-%m-%d"))

    # Return in chronological order (oldest first)
    return sorted(dates)


def backfill_snapshots(weeks=8, dry_run=False):
    """Create historical snapshots for trend data.

    Args:
        weeks: Number of weeks to backfill (default 8)
        dry_run: If True, don't actually insert snapshots

    Returns:
        Tuple of (successful_count, failed_count)
    """
    print("=" * 70)
    print("Historical Snapshot Backfiller")
    print("=" * 70)

    # Get date range from existing studies
    earliest, latest = get_study_date_range()

    if not earliest or not latest:
        print("\n✗ No studies found in database")
        return 0, 1

    print(f"\nStudy date range: {earliest} to {latest}")
    print(f"Backfilling last {weeks} weeks...")

    if dry_run:
        print("[DRY RUN MODE - No snapshots will be inserted]\n")

    # Generate weekly dates
    dates = generate_weekly_dates(latest, weeks)

    print(f"\nGenerating snapshots for {len(dates)} dates:")
    print("-" * 70)

    successful = 0
    failed = 0

    for snapshot_date in dates:
        try:
            # Only proceed if date is not too far in future
            if datetime.strptime(snapshot_date, "%Y-%m-%d") > datetime.now():
                print(f"  ⊘ {snapshot_date}: Skipping future date")
                continue

            if dry_run:
                print(f"  → {snapshot_date}: [DRY RUN - would create snapshot]")
            else:
                snapshot = create_snapshot_with_date(snapshot_date)
                if snapshot and snapshot.get("total_studies", 0) > 0:
                    print(
                        f"  ✓ {snapshot_date}: "
                        f"{snapshot['total_studies']} studies, "
                        f"{snapshot['studies_with_sex_metadata']} with metadata"
                    )
                    successful += 1
                else:
                    print(f"  ⊘ {snapshot_date}: No studies found for date")

        except Exception as e:
            logger.exception(f"Error creating snapshot for {snapshot_date}: {e}")
            print(f"  ✗ {snapshot_date}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    if dry_run:
        print("[DRY RUN] Would have created:")
    else:
        print(f"✓ Successfully created {successful} snapshots")

    if failed > 0:
        print(f"✗ Failed: {failed}")

    print("=" * 70)

    return successful, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=8,
        help="Number of weeks to backfill (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test without inserting snapshots",
    )

    args = parser.parse_args()

    try:
        successful, failed = backfill_snapshots(weeks=args.weeks, dry_run=args.dry_run)
        return 0 if failed == 0 else 1

    except KeyError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nMake sure Supabase secrets are configured:")
        print("  connections.supabase.SUPABASE_URL")
        print("  connections.supabase.SUPABASE_KEY")
        return 1

    except Exception as e:
        logger.exception(f"Error during backfill: {e}")
        print(f"\n✗ Error during backfill: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
