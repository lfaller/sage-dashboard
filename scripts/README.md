# SAGE Dashboard Scripts

Utility scripts for seeding and managing test data.

## Setup

All scripts require your Supabase credentials to be configured in `.streamlit/secrets.toml`:

```toml
[connections.supabase]
SUPABASE_URL = "your-project-url"
SUPABASE_KEY = "your-anon-key"
```

## Scripts

### 1. `load_sample_data.py`

Load sample study data into the `studies` table for testing.

**Usage:**
```bash
poetry run python scripts/load_sample_data.py
```

**What it does:**
- Creates 10 sample studies (human + mouse)
- Includes realistic metadata (GEO accessions, organism, study type, etc.)
- Sets sex metadata completeness and inferrability scores
- Uploads to Supabase `studies` table

**Output:**
- Study summary with counts
- Success/failure messages
- Error details if upload fails

### 2. `seed_disease_mappings.py`

Create disease mappings linking studies to diseases with clinical metadata.

**Prerequisites:**
- Must run `load_sample_data.py` first (requires studies to exist)

**Usage:**
```bash
poetry run python scripts/seed_disease_mappings.py
```

**What it does:**
- Creates 13 disease mappings across 10 studies
- Includes 11 unique disease terms across 5 categories:
  - **Cancer**: breast cancer, prostate cancer
  - **Cardiovascular**: heart failure, atherosclerosis
  - **Autoimmune**: systemic lupus erythematosus, rheumatoid arthritis
  - **Metabolic**: type 2 diabetes mellitus
  - **Psychiatric**: depression
  - **Infectious**: infection
- Includes clinical metadata:
  - Known sex difference flags
  - Sex bias direction (male/female)
  - Clinical priority scores (0.65-0.95)

**Output:**
- Mapping summary with disease counts
- Categories present in database
- Success/failure messages

## Workflow

To set up a complete testing environment:

```bash
# 1. Load sample studies first
poetry run python scripts/load_sample_data.py

# 2. Then seed disease mappings
poetry run python scripts/seed_disease_mappings.py

# 3. Start the app and test Disease Explorer
poetry run streamlit run app.py
```

Then navigate to the **Disease Explorer** page in the sidebar to test:
- Header metrics
- Disease filtering (by category, min studies, known sex differences)
- Disease visualization
- Drill-down to view studies per disease

## Resetting Data

To clear and reload data, you can:

1. Use Supabase SQL Editor to clear tables:
```sql
DELETE FROM disease_mappings;
DELETE FROM studies;
```

2. Re-run the seed scripts in order

## Notes

- Study IDs in `seed_disease_mappings.py` are hardcoded to match the 10 sample studies
- If you load different studies, you may need to update the study_id references
- All DOID (Disease Ontology) IDs are realistic references
- Clinical priority scores are estimated for demonstration purposes
