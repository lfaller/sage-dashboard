## SAGE - Sex-Aware Genomics Explorer

### The Problem

Sex differences impact ~12.5% of human gene expression, yet:

* Over 25% of critical metadata is missing from public genomics studies
* ~70% of studies claiming sex differences never actually test them statistically
* Failure to stratify by sex conceals up to 22% of significant expression differences
* Women experience adverse drug reactions nearly twice as often as men—partly due to this data gap

### The Solution

A free, open-source dashboard that:

* Assesses metadata completeness for sex across GEO/SRA datasets
* Identifies datasets where sex can be computationally inferred from expression data
* Prioritizes opportunities for reanalysis by disease relevance and clinical impact
* Tracks progress over time to measure community improvement

---

## Quick Start

### Run the Dashboard

```bash
poetry install
poetry run streamlit run app.py
```

Visit: http://localhost:8501

### Run Tests

```bash
poetry run pytest -v
```

---

## Deployment

Deploy to **Streamlit Community Cloud** (free, serverless):

1. **Push to GitHub** - Commit and push all changes to your repo
2. **Connect to Streamlit Cloud** - Go to [streamlit.io/cloud](https://streamlit.io/cloud) → "New app" → select your repo
3. **Add Secrets** - In your app's settings, add to the "Secrets" section:
   ```toml
   [connections.supabase]
   SUPABASE_URL = "your-project-url"
   SUPABASE_KEY = "your-anon-key"
   ```
4. **Done** - App auto-deploys when you push to main

Your app is now live and accessible to anyone with the URL!

---

## Development

- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference for common commands
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Full development guide with TDD workflow
- **[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)** - Contribution guidelines

### Technology Stack

- **Frontend:** Streamlit (Python-native dashboard)
- **Backend:** Python 3.11+
- **Database:** Supabase (PostgreSQL) with REST API
- **Data Source:** NCBI GEO (Gene Expression Omnibus) via GEOparse
- **Testing:** pytest (301 tests, 83% coverage)
- **Code Quality:** Black, Ruff
- **Dependency Management:** Poetry
- **CI/CD:** GitHub Actions with auto-formatting

---

## Data Sources

SAGE Dashboard analyzes metadata from real genomic research studies:

**NCBI Gene Expression Omnibus (GEO)**
- Public repository of ~200,000 genomics datasets
- Includes RNA-seq and microarray studies
- **Current focus:** Human studies, RNA-seq (ideal for sex inference)

**Data Loading**

Option 1: **Curated List** (Default)
```bash
# Load curated set of human RNA-seq studies
poetry run python scripts/fetch_geo_studies.py --limit 50

# Test with dry-run (no database writes)
poetry run python scripts/fetch_geo_studies.py --dry-run --limit 5

# Resume/update (skip existing studies)
poetry run python scripts/fetch_geo_studies.py --skip-existing --limit 200
```

Option 2: **Automated Search** (New! Uses NCBI Entrez API)
```bash
# Search last 5 years of human RNA-seq studies (100 studies max)
poetry run python scripts/fetch_geo_studies.py --search --limit 100

# Search last 2 years of mouse RNA-seq studies
poetry run python scripts/fetch_geo_studies.py --search \
    --organism "Mus musculus" --years-back 2 --limit 50

# Dry run to test
poetry run python scripts/fetch_geo_studies.py --search --limit 10 --dry-run
```

See [scripts/fetch_geo_studies.py](scripts/fetch_geo_studies.py) for details.

---

## Model Training

SAGE includes a complete workflow for training and validating sex classification models using elastic net logistic regression.

**Quick Start:**

```bash
# Train model using fixture data
poetry run python scripts/train_sex_classifier.py --fixture tiny

# Validate against test set
poetry run python scripts/validate_model.py --model sex_classifier --version 1.0.0 --test-data data.csv

# Check model status
poetry run python scripts/model_status.py --list
poetry run python scripts/model_status.py --model sex_classifier --version 1.0.0
```

**Features:**
- Elastic net logistic regression with cross-validation
- Training fixtures for reproducible tests (tiny, v1, imbalanced)
- Model versioning with semantic versioning (1.0.0, 2.0.0, etc.)
- Validation against Flynn et al. benchmarks
- Filesystem storage with JSON metadata and pickle models
- Database integration for audit trails and model history

See [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) for comprehensive guide.