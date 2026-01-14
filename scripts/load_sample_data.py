#!/usr/bin/env python3
"""Load sample study data into Supabase.

This script creates synthetic but realistic study data for testing/demo purposes.
In production, this would fetch from GEO/SRA APIs.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.data_loader import (  # noqa: E402
    Study,
    estimate_sex_inferrability,
    upsert_studies,
)


def create_sample_studies() -> list[Study]:
    """Create sample study data for demonstration.

    NOTE: These are synthetic study records created for testing/demo purposes.
    The accession IDs do NOT correspond to real GEO studies. Using a special
    prefix (DEMO-) to clearly distinguish from real data.
    """
    studies = [
        # Human RNA-seq studies
        Study(
            geo_accession="DEMO-001",
            title="Breast cancer transcriptome profiling",
            summary="RNA-seq analysis of breast cancer samples",
            organism="Homo sapiens",
            platform="Illumina HiSeq",
            study_type="RNA-seq",
            sample_count=156,
            has_sex_metadata=True,
            sex_metadata_completeness=0.92,
        ),
        Study(
            geo_accession="DEMO-002",
            title="Cardiac remodeling in heart failure",
            organism="Homo sapiens",
            platform="Illumina NovaSeq",
            study_type="RNA-seq",
            sample_count=89,
            has_sex_metadata=False,
            sex_metadata_completeness=0.0,
        ),
        Study(
            geo_accession="DEMO-003",
            title="Systemic lupus erythematosus patient cohort",
            summary="Transcriptome study of SLE patients vs controls",
            organism="Homo sapiens",
            platform="Illumina HiSeq",
            study_type="RNA-seq",
            sample_count=234,
            has_sex_metadata=True,
            sex_metadata_completeness=0.78,
        ),
        Study(
            geo_accession="DEMO-004",
            title="Type 2 diabetes gene expression analysis",
            organism="Homo sapiens",
            platform="Affymetrix",
            study_type="microarray",
            sample_count=145,
            has_sex_metadata=True,
            sex_metadata_completeness=0.85,
        ),
        Study(
            geo_accession="DEMO-005",
            title="Rheumatoid arthritis synovial tissue",
            organism="Homo sapiens",
            platform="Illumina HiSeq",
            study_type="RNA-seq",
            sample_count=78,
            has_sex_metadata=False,
            sex_metadata_completeness=0.0,
        ),
        # Mouse studies
        Study(
            geo_accession="DEMO-006",
            title="Immunological response in mice",
            organism="Mus musculus",
            platform="Illumina HiSeq",
            study_type="RNA-seq",
            sample_count=64,
            has_sex_metadata=True,
            sex_metadata_completeness=1.0,
        ),
        Study(
            geo_accession="DEMO-007",
            title="Mouse kidney transcriptomics",
            organism="Mus musculus",
            platform="Agilent",
            study_type="microarray",
            sample_count=48,
            has_sex_metadata=False,
            sex_metadata_completeness=0.0,
        ),
        # Additional human studies
        Study(
            geo_accession="DEMO-008",
            title="Depression biomarkers in blood",
            organism="Homo sapiens",
            platform="Illumina NovaSeq",
            study_type="RNA-seq",
            sample_count=312,
            has_sex_metadata=True,
            sex_metadata_completeness=0.67,
        ),
        Study(
            geo_accession="DEMO-009",
            title="Prostate cancer methylation analysis",
            organism="Homo sapiens",
            platform="Illumina HiSeq",
            study_type="RNA-seq",
            sample_count=203,
            has_sex_metadata=True,
            sex_metadata_completeness=0.95,
        ),
        Study(
            geo_accession="DEMO-010",
            title="Atherosclerosis risk markers",
            organism="Homo sapiens",
            platform="Illumina HiSeq",
            study_type="RNA-seq",
            sample_count=127,
            has_sex_metadata=False,
            sex_metadata_completeness=0.0,
        ),
    ]

    # Calculate sex inferrability for RNA-seq studies
    for study in studies:
        if study.study_type == "RNA-seq":
            study.sex_inferrable = estimate_sex_inferrability(study)
            if study.sex_inferrable:
                study.sex_inference_confidence = 0.85

    return studies


def main():
    """Load sample data into Supabase."""
    print("Creating sample studies...")
    studies = create_sample_studies()
    print(f"Created {len(studies)} sample studies")

    print("\nStudy summary:")
    print(f"  Total studies: {len(studies)}")
    print(f"  With sex metadata: {sum(1 for s in studies if s.has_sex_metadata)}")
    print(f"  RNA-seq studies: {sum(1 for s in studies if s.study_type == 'RNA-seq')}")
    print(f"  Inferrable studies: {sum(1 for s in studies if s.sex_inferrable)}")

    print("\nSample studies:")
    for study in studies[:3]:
        print(f"  - {study.geo_accession}: {study.title}")
        print(f"    Organism: {study.organism}, Samples: {study.sample_count}")
        print(
            f"    Sex metadata: {study.sex_metadata_completeness:.1%}, "
            f"Inferrable: {study.sex_inferrable}"
        )

    print("\nUploading to Supabase...")
    try:
        result = upsert_studies(studies)
        print(f"Successfully uploaded {len(result)} studies!")
        print("Sample data loaded successfully.")
        return 0
    except Exception as e:
        print(f"Error uploading to Supabase: {e}", file=sys.stderr)
        print("Make sure your .streamlit/secrets.toml is configured correctly.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
