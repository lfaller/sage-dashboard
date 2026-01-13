# Quick Start Guide

## Setup (First Time)

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

## Development Commands

### Run Tests
```bash
poetry run pytest           # Run all tests
poetry run pytest -v        # Verbose output
poetry run pytest --cov     # With coverage report
```

### Run Dashboard
```bash
poetry run streamlit run app.py
```
Then open: http://localhost:8501

### Code Quality
```bash
poetry run black .          # Format code
poetry run ruff check .     # Lint code
poetry run ruff check . --fix  # Auto-fix lint issues
```

### Add a New Dependency
```bash
poetry add package-name           # Add to main dependencies
poetry add --group dev package-name  # Add to dev dependencies
```

---

## Project Status

✅ **MVP Complete**
- Streamlit app with home page & overview dashboard
- Mock data visualization
- Unit tests with 100% coverage
- Poetry dependency management
- TDD-ready structure

**Current files:**
- `app.py` - Main entry point
- `pages/1_📊_Overview.py` - Overview dashboard
- `src/sage/metrics.py` - Core metrics module
- `tests/test_metrics.py` - Unit tests
- `pyproject.toml` - Poetry config

---

## Next Steps

1. Connect to Supabase database
2. Load real study data from GEO/SRA
3. Add disease mapping functionality
4. Create study search page
5. Build rescue finder with ROI scoring
