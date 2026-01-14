#!/usr/bin/env python3
"""Seed disease mappings into Supabase for testing Disease Explorer UI.

This script creates sample disease mappings linking studies to diseases
with clinical relevance data for demonstration purposes.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.database import get_supabase_client  # noqa: E402


def seed_disease_mappings():
    """Create and insert sample disease mappings."""
    client = get_supabase_client()

    # Sample disease mappings linking studies to diseases
    disease_mappings = [
        # Breast cancer mappings
        {
            "study_id": 1,  # GSE123001 - Breast cancer transcriptome
            "disease_term": "breast cancer",
            "doid_id": "DOID:1612",
            "doid_name": "breast cancer",
            "disease_category": "cancer",
            "known_sex_difference": True,
            "sex_bias_direction": "female",
            "clinical_priority_score": 0.95,
        },
        {
            "study_id": 9,  # GSE123009 - Prostate cancer
            "disease_term": "breast cancer",
            "doid_id": "DOID:1612",
            "doid_name": "breast cancer",
            "disease_category": "cancer",
            "known_sex_difference": True,
            "sex_bias_direction": "female",
            "clinical_priority_score": 0.92,
        },
        # Heart failure mappings
        {
            "study_id": 2,  # GSE123002 - Cardiac remodeling
            "disease_term": "heart failure",
            "doid_id": "DOID:6000",
            "doid_name": "heart failure",
            "disease_category": "cardiovascular",
            "known_sex_difference": True,
            "sex_bias_direction": "male",
            "clinical_priority_score": 0.88,
        },
        {
            "study_id": 10,  # GSE123010 - Atherosclerosis
            "disease_term": "heart failure",
            "doid_id": "DOID:6000",
            "doid_name": "heart failure",
            "disease_category": "cardiovascular",
            "known_sex_difference": True,
            "sex_bias_direction": "male",
            "clinical_priority_score": 0.85,
        },
        # Systemic lupus erythematosus (SLE)
        {
            "study_id": 3,  # GSE123003 - SLE patient cohort
            "disease_term": "systemic lupus erythematosus",
            "doid_id": "DOID:9074",
            "doid_name": "systemic lupus erythematosus",
            "disease_category": "autoimmune",
            "known_sex_difference": True,
            "sex_bias_direction": "female",
            "clinical_priority_score": 0.90,
        },
        # Type 2 diabetes
        {
            "study_id": 4,  # GSE123004 - Type 2 diabetes
            "disease_term": "type 2 diabetes mellitus",
            "doid_id": "DOID:9352",
            "doid_name": "type 2 diabetes mellitus",
            "disease_category": "metabolic",
            "known_sex_difference": True,
            "sex_bias_direction": "female",
            "clinical_priority_score": 0.87,
        },
        {
            "study_id": 8,  # GSE123008 - Depression biomarkers
            "disease_term": "type 2 diabetes mellitus",
            "doid_id": "DOID:9352",
            "doid_name": "type 2 diabetes mellitus",
            "disease_category": "metabolic",
            "known_sex_difference": True,
            "sex_bias_direction": "female",
            "clinical_priority_score": 0.84,
        },
        # Rheumatoid arthritis
        {
            "study_id": 5,  # GSE123005 - Rheumatoid arthritis
            "disease_term": "rheumatoid arthritis",
            "doid_id": "DOID:7148",
            "doid_name": "rheumatoid arthritis",
            "disease_category": "autoimmune",
            "known_sex_difference": True,
            "sex_bias_direction": "female",
            "clinical_priority_score": 0.86,
        },
        # Depression
        {
            "study_id": 8,  # GSE123008 - Depression biomarkers
            "disease_term": "depression",
            "doid_id": "DOID:1470",
            "doid_name": "major depressive disorder",
            "disease_category": "psychiatric",
            "known_sex_difference": True,
            "sex_bias_direction": "female",
            "clinical_priority_score": 0.91,
        },
        # Prostate cancer
        {
            "study_id": 9,  # GSE123009 - Prostate cancer methylation
            "disease_term": "prostate cancer",
            "doid_id": "DOID:2994",
            "doid_name": "prostate cancer",
            "disease_category": "cancer",
            "known_sex_difference": True,
            "sex_bias_direction": "male",
            "clinical_priority_score": 0.93,
        },
        # Atherosclerosis
        {
            "study_id": 10,  # GSE123010 - Atherosclerosis risk
            "disease_term": "atherosclerosis",
            "doid_id": "DOID:1936",
            "doid_name": "atherosclerosis",
            "disease_category": "cardiovascular",
            "known_sex_difference": True,
            "sex_bias_direction": "male",
            "clinical_priority_score": 0.82,
        },
        # Mouse immune response study
        {
            "study_id": 6,  # GSE123006 - Mouse immune response
            "disease_term": "infection",
            "doid_id": "DOID:0050117",
            "doid_name": "infection",
            "disease_category": "infectious",
            "known_sex_difference": False,
            "sex_bias_direction": None,
            "clinical_priority_score": 0.65,
        },
    ]

    print(f"Seeding {len(disease_mappings)} disease mappings...")

    try:
        # Insert disease mappings
        response = client.table("disease_mappings").insert(disease_mappings).execute()

        if response.data:
            print(f"✓ Successfully inserted {len(response.data)} disease mappings")

            # Print summary
            diseases = set(d["disease_term"] for d in disease_mappings)
            categories = set(d["disease_category"] for d in disease_mappings)

            print("\nDisease Mapping Summary:")
            print(f"  Total mappings: {len(disease_mappings)}")
            print(f"  Unique diseases: {len(diseases)}")
            print(f"  Disease categories: {sorted(categories)}")
            print("\nDiseases:")
            for disease in sorted(diseases):
                count = sum(1 for d in disease_mappings if d["disease_term"] == disease)
                print(f"  - {disease}: {count} studies")

            return 0
        else:
            print("✗ No data returned from insert")
            return 1

    except Exception as e:
        print(f"✗ Error seeding disease mappings: {e}", file=sys.stderr)
        print("Make sure:", file=sys.stderr)
        print("  1. Your .streamlit/secrets.toml is configured", file=sys.stderr)
        print(
            "  2. Sample studies have been loaded (run load_sample_data.py first)", file=sys.stderr
        )
        print("  3. Disease_mappings table exists in Supabase", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(seed_disease_mappings())
