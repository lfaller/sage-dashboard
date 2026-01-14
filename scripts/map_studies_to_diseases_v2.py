"""Map studies to diseases using curated disease-accession mappings.

This script uses the pre-defined disease mappings from the curated study list
and maps studies to diseases based on their GEO accessions.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sage.database import get_supabase_client  # noqa: E402
from sage.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


# Curated disease-study mappings based on GEO accessions
ACCESSION_TO_DISEASES = {
    # Breast Cancer Studies
    "GSE175343": ["breast cancer"],
    "GSE176043": ["breast cancer"],
    "GSE180322": ["breast cancer"],
    "GSE184401": ["breast cancer"],
    "GSE188753": ["breast cancer"],
    # Lung Cancer Studies
    "GSE168204": ["lung cancer"],
    "GSE169338": ["lung cancer"],
    "GSE175145": ["lung cancer"],
    "GSE188373": ["lung cancer"],
    "GSE178220": ["lung cancer"],
    # Prostate Cancer Studies
    "GSE180270": ["prostate cancer"],
    "GSE163917": ["prostate cancer"],
    "GSE180573": ["prostate cancer"],
    "GSE169038": ["prostate cancer"],
    "GSE176911": ["prostate cancer"],
    # Colorectal Cancer Studies
    "GSE185191": ["colorectal cancer"],
    "GSE178341": ["colorectal cancer"],
    "GSE165490": ["colorectal cancer"],
    "GSE177651": ["colorectal cancer"],
    "GSE172457": ["colorectal cancer"],
    # Ovarian Cancer Studies
    "GSE184234": ["ovarian cancer"],
    "GSE181552": ["ovarian cancer"],
    "GSE170826": ["ovarian cancer"],
    "GSE177843": ["ovarian cancer"],
    "GSE157421": ["ovarian cancer"],
    # Melanoma Studies
    "GSE185060": ["melanoma"],
    "GSE172857": ["melanoma"],
    "GSE175281": ["melanoma"],
    "GSE178882": ["melanoma"],
    "GSE166742": ["melanoma"],
    # Cardiovascular Studies
    "GSE179588": ["heart failure"],
    "GSE183220": ["coronary artery disease"],
    "GSE175545": ["heart failure"],
    "GSE168649": ["heart failure"],
    "GSE180334": ["arrhythmia"],
    # Neurological Studies
    "GSE174576": ["alzheimer's disease"],
    "GSE180316": ["parkinson's disease"],
    "GSE170844": ["amyotrophic lateral sclerosis"],
    "GSE178244": ["multiple sclerosis"],
    "GSE169442": ["dementia"],
    # Metabolic Studies
    "GSE183045": ["type 2 diabetes mellitus"],
    "GSE178231": ["obesity"],
    "GSE175673": ["nafld"],
    "GSE179056": ["metabolic syndrome"],
    "GSE181220": ["dyslipidemia"],
    # Immune/Infectious Disease Studies
    "GSE181426": ["covid-19"],
    "GSE173363": ["sepsis"],
    "GSE177892": ["hiv infection"],
    "GSE174568": ["influenza infection"],
    "GSE180445": ["tuberculosis"],
    # Rare Disease Studies
    "GSE176805": ["lysosomal storage disease"],
    "GSE178455": ["immunodeficiency"],
    "GSE180698": ["skeletal dysplasia"],
    "GSE171987": ["mitochondrial disease"],
    "GSE182333": ["ciliopathy"],
}


# Disease metadata
DISEASE_METADATA = {
    "breast cancer": {
        "doid_id": "DOID:1612",
        "doid_name": "breast cancer",
        "disease_category": "cancer",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.95,
    },
    "lung cancer": {
        "doid_id": "DOID:1324",
        "doid_name": "lung cancer",
        "disease_category": "cancer",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.92,
    },
    "prostate cancer": {
        "doid_id": "DOID:2994",
        "doid_name": "prostate cancer",
        "disease_category": "cancer",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.93,
    },
    "colorectal cancer": {
        "doid_id": "DOID:9256",
        "doid_name": "colorectal cancer",
        "disease_category": "cancer",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.90,
    },
    "ovarian cancer": {
        "doid_id": "DOID:2394",
        "doid_name": "ovarian cancer",
        "disease_category": "cancer",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.94,
    },
    "melanoma": {
        "doid_id": "DOID:1909",
        "doid_name": "melanoma",
        "disease_category": "cancer",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.89,
    },
    "heart failure": {
        "doid_id": "DOID:6000",
        "doid_name": "heart failure",
        "disease_category": "cardiovascular",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.88,
    },
    "coronary artery disease": {
        "doid_id": "DOID:5844",
        "doid_name": "coronary artery disease",
        "disease_category": "cardiovascular",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.86,
    },
    "arrhythmia": {
        "doid_id": "DOID:0050417",
        "doid_name": "arrhythmia",
        "disease_category": "cardiovascular",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.82,
    },
    "alzheimer's disease": {
        "doid_id": "DOID:10652",
        "doid_name": "Alzheimer's disease",
        "disease_category": "neurological",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.91,
    },
    "parkinson's disease": {
        "doid_id": "DOID:14330",
        "doid_name": "Parkinson's disease",
        "disease_category": "neurological",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.89,
    },
    "amyotrophic lateral sclerosis": {
        "doid_id": "DOID:332",
        "doid_name": "amyotrophic lateral sclerosis",
        "disease_category": "neurological",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.88,
    },
    "multiple sclerosis": {
        "doid_id": "DOID:2377",
        "doid_name": "multiple sclerosis",
        "disease_category": "neurological",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.87,
    },
    "dementia": {
        "doid_id": "DOID:1816",
        "doid_name": "dementia",
        "disease_category": "neurological",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.85,
    },
    "type 2 diabetes mellitus": {
        "doid_id": "DOID:9352",
        "doid_name": "type 2 diabetes mellitus",
        "disease_category": "metabolic",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.85,
    },
    "obesity": {
        "doid_id": "DOID:9970",
        "doid_name": "obesity",
        "disease_category": "metabolic",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.83,
    },
    "nafld": {
        "doid_id": "DOID:0080547",
        "doid_name": "non-alcoholic fatty liver disease",
        "disease_category": "metabolic",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.81,
    },
    "metabolic syndrome": {
        "doid_id": "DOID:4136",
        "doid_name": "metabolic syndrome",
        "disease_category": "metabolic",
        "known_sex_difference": True,
        "sex_bias_direction": "female",
        "clinical_priority_score": 0.80,
    },
    "dyslipidemia": {
        "doid_id": "DOID:1816",
        "doid_name": "dyslipidemia",
        "disease_category": "metabolic",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.78,
    },
    "covid-19": {
        "doid_id": "DOID:0080600",
        "doid_name": "COVID-19",
        "disease_category": "infectious",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.88,
    },
    "sepsis": {
        "doid_id": "DOID:11065",
        "doid_name": "sepsis",
        "disease_category": "infectious",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.84,
    },
    "hiv infection": {
        "doid_id": "DOID:526",
        "doid_name": "HIV infection",
        "disease_category": "infectious",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.83,
    },
    "influenza infection": {
        "doid_id": "DOID:8469",
        "doid_name": "influenza",
        "disease_category": "infectious",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.79,
    },
    "tuberculosis": {
        "doid_id": "DOID:399",
        "doid_name": "tuberculosis",
        "disease_category": "infectious",
        "known_sex_difference": True,
        "sex_bias_direction": "male",
        "clinical_priority_score": 0.82,
    },
    "lysosomal storage disease": {
        "doid_id": "DOID:28",
        "doid_name": "lysosomal storage disease",
        "disease_category": "genetic",
        "known_sex_difference": False,
        "sex_bias_direction": None,
        "clinical_priority_score": 0.75,
    },
    "immunodeficiency": {
        "doid_id": "DOID:682",
        "doid_name": "immunodeficiency",
        "disease_category": "genetic",
        "known_sex_difference": False,
        "sex_bias_direction": None,
        "clinical_priority_score": 0.80,
    },
    "skeletal dysplasia": {
        "doid_id": "DOID:0090026",
        "doid_name": "skeletal dysplasia",
        "disease_category": "genetic",
        "known_sex_difference": False,
        "sex_bias_direction": None,
        "clinical_priority_score": 0.72,
    },
    "mitochondrial disease": {
        "doid_id": "DOID:0090063",
        "doid_name": "mitochondrial disease",
        "disease_category": "genetic",
        "known_sex_difference": False,
        "sex_bias_direction": None,
        "clinical_priority_score": 0.76,
    },
    "ciliopathy": {
        "doid_id": "DOID:0090063",
        "doid_name": "ciliopathy",
        "disease_category": "genetic",
        "known_sex_difference": False,
        "sex_bias_direction": None,
        "clinical_priority_score": 0.71,
    },
}


def main():
    """Map studies to diseases using accession lookup."""
    print("=" * 70)
    print("Study-to-Disease Mapping (Curated)")
    print("=" * 70)

    client = get_supabase_client()

    try:
        # Fetch all studies
        print("\nFetching studies from database...")
        studies_response = client.table("studies").select("id, geo_accession").execute()
        studies = studies_response.data or []

        if not studies:
            print("No studies found in database. Load studies first.")
            return 1

        print(f"Found {len(studies)} studies")

        # Map diseases based on accession lookup
        disease_mappings = []
        disease_set = set()
        mapped_count = 0

        print("\nMatching studies to diseases...")
        for study in studies:
            study_id = study["id"]
            accession = study.get("geo_accession", "")

            if accession in ACCESSION_TO_DISEASES:
                diseases = ACCESSION_TO_DISEASES[accession]
                for disease_name in diseases:
                    if disease_name in DISEASE_METADATA:
                        disease_info = DISEASE_METADATA[disease_name]
                        mapping = {
                            "study_id": study_id,
                            "disease_term": disease_name,
                            "doid_id": disease_info["doid_id"],
                            "doid_name": disease_info["doid_name"],
                            "disease_category": disease_info["disease_category"],
                            "known_sex_difference": disease_info["known_sex_difference"],
                            "sex_bias_direction": disease_info["sex_bias_direction"],
                            "clinical_priority_score": disease_info["clinical_priority_score"],
                        }
                        disease_mappings.append(mapping)
                        disease_set.add(disease_name)
                        mapped_count += 1

        if not disease_mappings:
            print("No diseases mapped. Check GEO accessions in database.")
            return 1

        print(f"Mapped {mapped_count} disease associations")
        print(f"Unique diseases: {len(disease_set)}")

        # Clear existing mappings
        print("\nClearing existing disease mappings...")
        client.table("disease_mappings").delete().gt("id", 0).execute()

        # Insert new mappings
        print(f"Inserting {len(disease_mappings)} disease mappings...")
        response = client.table("disease_mappings").insert(disease_mappings).execute()

        if response.data:
            print(f"✓ Successfully inserted {len(response.data)} disease mappings")

            # Print summary
            print("\nDisease Mapping Summary:")
            print(f"  Total mappings: {len(disease_mappings)}")
            print(f"  Unique diseases: {len(disease_set)}")

            categories = sorted(set(m["disease_category"] for m in disease_mappings))
            print(f"  Categories: {categories}")

            print("\nTop diseases by study count:")
            disease_counts = {}
            for mapping in disease_mappings:
                disease = mapping["disease_term"]
                disease_counts[disease] = disease_counts.get(disease, 0) + 1

            for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[
                :15
            ]:
                print(f"  - {disease}: {count} studies")

            return 0
        else:
            print("✗ No data returned from insert")
            return 1

    except Exception as e:
        logger.exception(f"Error mapping studies to diseases: {e}")
        print(f"\n✗ Error mapping studies to diseases: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
