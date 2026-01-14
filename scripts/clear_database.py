"""Clear all studies from the database to prepare for fresh data load."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.database import get_supabase_client  # noqa: E402
from sage.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    """Clear all studies from database."""
    print("=" * 70)
    print("Database Cleanup Utility")
    print("=" * 70)

    client = get_supabase_client()

    # Confirm before deleting
    response = input("\nThis will DELETE ALL studies from the database. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return 0

    try:
        # Get count before deletion
        count_response = client.table("studies").select("id", count="exact").execute()
        count_before = count_response.count or 0

        print(f"\nDeleting {count_before} studies...")

        # Delete all studies
        client.table("studies").delete().neq("id", "").execute()

        print("✓ Deleted successfully!")
        print("\nDatabase is now empty. Ready for fresh data load.")
        print("\nNext step: Run")
        print("  poetry run python scripts/fetch_geo_studies.py --limit 50")

    except Exception as e:
        logger.exception(f"Error clearing database: {e}")
        print(f"\n✗ Error clearing database: {e}", file=sys.stderr)
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
