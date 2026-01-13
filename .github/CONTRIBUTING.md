# Contributing to SAGE Dashboard

Thank you for contributing to the SAGE project! This guide will help you set up your development environment and follow our contribution guidelines.

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/linafaller/sage-dashboard.git
cd sage-dashboard
poetry install
```

### 2. Pre-commit Hooks

The project uses Git pre-commit hooks to automatically format and lint code. These are already configured in `.git/hooks/pre-commit`.

When you commit, the hook will:
- Run `black` to format your code
- Run `ruff check --fix` to lint and auto-fix issues
- Re-stage any changes made by the tools

**First commit after cloning will automatically set this up.**

## Development Workflow

### Writing Code (TDD Approach)

1. **Write a test first** in `tests/`
   ```bash
   # Add your test to tests/test_*.py
   ```

2. **Run tests to confirm failure**
   ```bash
   poetry run pytest -v
   ```

3. **Implement code** to make the test pass
   ```bash
   # Add your implementation in src/sage/
   ```

4. **Run all tests**
   ```bash
   poetry run pytest -v --cov
   ```

### Code Quality

Before pushing, ensure:

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check . --fix

# Run all tests with coverage
poetry run pytest --cov=src
```

Or let the pre-commit hook do it automatically on `git commit`!

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `chore:` - Build, dependencies, etc.
- `test:` - Adding or updating tests
- `refactor:` - Code changes that don't add features or fix bugs

**Example:**
```
feat: add disease filtering to search page

- Implement disease field dropdown
- Add database query for disease terms
- Update tests for new filter logic

Co-Authored-By: Name <email@example.com>
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes and commit**
   - Pre-commit hook will format/lint automatically
   - Write tests first (TDD)
   - Keep commits logical and well-described

3. **Push and open a PR**
   ```bash
   git push -u origin feat/your-feature-name
   ```

4. **GitHub Actions will automatically**
   - Run Black format check
   - Run Ruff linting
   - Run all tests with coverage
   - Upload coverage to Codecov

5. **Address any feedback** and push updates

6. **Merge when approved!**

## Project Structure

```
sage-dashboard/
├── src/sage/                 # Main package code
│   ├── metrics.py           # Core metrics calculations
│   └── ...                  # Future modules
├── tests/                   # Unit tests
│   ├── test_metrics.py
│   └── ...
├── pages/                   # Streamlit pages
│   └── 1_📊_Overview.py
├── app.py                   # Streamlit main entry point
├── pyproject.toml           # Poetry config + tool settings
└── .github/
    └── workflows/           # GitHub Actions
```

## Testing

All code must be tested. We aim for high coverage:

```bash
# Run tests with coverage report
poetry run pytest --cov=src --cov-report=html

# View coverage (opens in browser)
open htmlcov/index.html
```

## Common Tasks

### Add a New Dependency

```bash
# Main dependency
poetry add package-name

# Dev dependency
poetry add --group dev package-name
```

### Run Dashboard Locally

```bash
poetry run streamlit run app.py
# Opens at http://localhost:8501
```

### Format All Code

```bash
poetry run black .
poetry run ruff check . --fix
```

## Questions?

Open an issue or check the existing documentation in `QUICKSTART.md` and `DEVELOPMENT.md`.

---

**Happy coding!** 🧬
