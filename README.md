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

## Development

- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference for common commands
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Full development guide with TDD workflow
- **[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)** - Contribution guidelines

### Technology Stack

- **Frontend:** Streamlit (Python-native dashboard)
- **Backend:** Python 3.11+
- **Database:** Supabase (PostgreSQL) with REST API
- **Testing:** pytest (55 tests, 88% coverage)
- **Code Quality:** Black, Ruff
- **Dependency Management:** Poetry
- **CI/CD:** GitHub Actions with auto-formatting