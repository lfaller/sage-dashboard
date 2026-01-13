# SAGE Dashboard - Development Setup

## Quick Start

### 1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Install Dependencies
```bash
poetry install
```

### 3. Run Tests
```bash
poetry run pytest
```

### 4. Run the Dashboard Locally
```bash
poetry run streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## Project Structure

```
sage-dashboard/
├── app.py                      # Main entry point
├── pages/
│   └── 1_📊_Overview.py        # Overview page
├── src/
│   └── sage/
│       ├── __init__.py
│       └── metrics.py          # Core metrics calculations
├── tests/
│   ├── __init__.py
│   └── test_metrics.py         # Unit tests
├── pyproject.toml              # Poetry configuration
├── .gitignore
└── README.md
```

---

## Development Workflow (TDD)

1. **Write a test first** in `tests/`
2. **Run tests** to confirm they fail: `poetry run pytest`
3. **Implement code** to make the test pass
4. **Refactor** and ensure all tests still pass

### Example:

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_metrics.py

# Run with verbose output
poetry run pytest -v
```

---

## Current MVP Features

- ✅ Minimal Streamlit app with home page
- ✅ Overview page with mock data
- ✅ Metrics calculations module (tested)
- ✅ Poetry for dependency management
- ✅ Pytest for test-driven development

---

## Next Steps

1. Add data loading functions (with tests)
2. Connect to Supabase database
3. Add study search functionality
4. Build disease explorer page
5. Add more complex visualizations

---

## Code Quality

- All code uses Poetry for dependency management
- Unit tests in `tests/` directory
- Test-driven development approach
- Code formatted with Black (configured in pyproject.toml)
- Linting with Ruff (configured in pyproject.toml)

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check . --fix
```

