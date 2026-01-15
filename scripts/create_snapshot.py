#!/usr/bin/env python3
"""Create weekly completeness snapshot for trend tracking."""
import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.database import get_supabase_client, create_snapshot  # noqa: E402
from sage.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def get_organism_options() -> list[str]:
    """Get list of organisms in database."""
    try:
        client = get_supabase_client()
        response = client.table("studies").select("organism").execute()
        organisms = list(set(item["organism"] for item in response.data or [] if item["organism"]))
        return sorted(organisms)
    except Exception as e:
        logger.error(f"Error fetching organisms: {e}")
        return []


def get_disease_category_options() -> list[str]:
    """Get list of disease categories in database."""
    try:
        client = get_supabase_client()
        response = client.table("disease_mappings").select("disease_category").execute()
        categories = list(
            set(
                item["disease_category"] for item in response.data or [] if item["disease_category"]
            )
        )
        return sorted(categories)
    except Exception as e:
        logger.error(f"Error fetching disease categories: {e}")
        return []


def main():
    """Create one or more completeness snapshots."""
    parser = argparse.ArgumentParser(
        description="Create completeness snapshots for trend tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create overall snapshot
  poetry run python scripts/create_snapshot.py

  # Create for specific organism
  poetry run python scripts/create_snapshot.py --organism "Homo sapiens"

  # Create for specific disease category
  poetry run python scripts/create_snapshot.py --disease-category "cancer"

  # Create snapshots for all organism+disease combinations
  poetry run python scripts/create_snapshot.py --all-combinations
        """,
    )
    parser.add_argument("--organism", help="Filter by organism (e.g., 'Homo sapiens')")
    parser.add_argument("--disease-category", help="Filter by disease category (e.g., 'cancer')")
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Create snapshots for all organism and disease category combinations",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Completeness Snapshot Creator")
    print("=" * 70)

    snapshots_created = []
    snapshots_failed = []

    try:
        if args.all_combinations:
            # Create snapshots for all combinations
            organisms = get_organism_options()
            categories = get_disease_category_options()

            print(
                f"\nCreating snapshots for {len(organisms)} organisms × {len(categories)} categories"
            )
            print(f"Total snapshots: {len(organisms) * len(categories)}\n")

            # Overall snapshot first
            print("Creating overall snapshot (all organisms, all diseases)...")
            try:
                snapshot = create_snapshot()
                snapshots_created.append(snapshot)
                print("  ✓ Overall snapshot created")
            except Exception as e:
                logger.exception(f"Error creating overall snapshot: {e}")
                snapshots_failed.append({"filters": "all", "error": str(e)})
                print(f"  ✗ Failed: {e}")

            # Organism-only snapshots
            for organism in organisms:
                try:
                    snapshot = create_snapshot(organism=organism)
                    snapshots_created.append(snapshot)
                    print(f"  ✓ {organism}")
                except Exception as e:
                    logger.exception(f"Error creating snapshot for {organism}: {e}")
                    snapshots_failed.append({"filters": f"organism={organism}", "error": str(e)})
                    print(f"  ✗ {organism}: {e}")

            # Disease-only snapshots
            for category in categories:
                try:
                    snapshot = create_snapshot(disease_category=category)
                    snapshots_created.append(snapshot)
                    print(f"  ✓ {category}")
                except Exception as e:
                    logger.exception(f"Error creating snapshot for {category}: {e}")
                    snapshots_failed.append(
                        {"filters": f"disease_category={category}", "error": str(e)}
                    )
                    print(f"  ✗ {category}: {e}")

            # Organism + disease combinations
            print(
                f"\nCreating {len(organisms)} × {len(categories)} organism-disease combinations..."
            )
            for organism in organisms:
                for category in categories:
                    try:
                        snapshot = create_snapshot(organism=organism, disease_category=category)
                        snapshots_created.append(snapshot)
                    except Exception as e:
                        logger.exception(
                            f"Error creating snapshot for {organism} + {category}: {e}"
                        )
                        snapshots_failed.append(
                            {
                                "filters": f"organism={organism}, disease_category={category}",
                                "error": str(e),
                            }
                        )

            print(f"  ✓ Created {len(organisms) * len(categories)} organization-disease snapshots")

        else:
            # Create single snapshot with specified filters
            if args.organism or args.disease_category:
                print("\nCreating snapshot with filters:")
                if args.organism:
                    print(f"  Organism: {args.organism}")
                if args.disease_category:
                    print(f"  Disease Category: {args.disease_category}")
            else:
                print("\nCreating overall snapshot (all organisms, all diseases)...")

            snapshot = create_snapshot(
                organism=args.organism, disease_category=args.disease_category
            )
            snapshots_created.append(snapshot)

            if snapshot and snapshot.get("total_studies", 0) > 0:
                print("\n✓ Snapshot created successfully!")
                print("\nSnapshot Summary:")
                print(f"  Date: {snapshot['snapshot_date']}")
                print(f"  Total Studies: {snapshot['total_studies']:,}")
                print(f"  With Sex Metadata: {snapshot['studies_with_sex_metadata']:,}")
                print(f"  Sex Inferrable: {snapshot['studies_sex_inferrable']:,}")
                print(f"  Analyzed by Sex: {snapshot['studies_with_sex_analysis']:,}")
                print(
                    f"  Avg Metadata Completeness: {snapshot['avg_metadata_completeness'] * 100:.1f}%"
                )
            else:
                print("\n⚠ Snapshot created but no studies found with specified filters")

    except KeyError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nMake sure Supabase secrets are configured:")
        print("  connections.supabase.SUPABASE_URL")
        print("  connections.supabase.SUPABASE_KEY")
        return 1

    except Exception as e:
        logger.exception(f"Error creating snapshot: {e}")
        print(f"\n✗ Error creating snapshot: {e}")
        return 1

    # Print summary
    print("\n" + "=" * 70)
    if snapshots_created:
        print(f"✓ Successfully created {len(snapshots_created)} snapshots")
    if snapshots_failed:
        print(f"✗ Failed to create {len(snapshots_failed)} snapshots:")
        for failed in snapshots_failed:
            print(f"  - {failed['filters']}: {failed['error']}")
    print("=" * 70)

    return 0 if not snapshots_failed else 1


if __name__ == "__main__":
    sys.exit(main())
