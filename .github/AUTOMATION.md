# Automated Code Quality Setup

This project uses two complementary automation systems to ensure code quality: **local Git hooks** and **GitHub Actions**.

## Local Development: Git Pre-Commit Hook

### What It Does

The `.git/hooks/pre-commit` script runs **before every commit** and automatically:

1. **Formats code with Black** - Ensures consistent style
2. **Lints and fixes with Ruff** - Catches and auto-fixes issues
3. **Re-stages changes** - Includes any formatting changes in the commit
4. **Fails the commit if errors occur** - Prevents bad code from being committed

### When It Runs

```bash
$ git add src/sage/my_new_feature.py
$ git commit -m "feat: add new feature"

# Hook automatically runs:
Running pre-commit hooks...
Checking files:
src/sage/my_new_feature.py

Running Black...
✓ Black passed

Running Ruff...
✓ Ruff passed

✓ All pre-commit hooks passed
```

### What Files Are Checked

Only **Python files that are staged** are checked. Other files (markdown, YAML, etc.) are ignored.

### If Formatting Changes Are Made

The hook automatically re-stages the changes so they're included in your commit:

```
# Your code before:
def my_function(  x,y  ):  return x+y

# After Black+Ruff:
def my_function(x, y):
    return x + y

# Automatically re-staged and included in your commit!
```

### Override (Not Recommended)

If you need to skip the hook (not recommended):

```bash
git commit --no-verify
```

## Remote CI/CD: GitHub Actions

### What It Does

The `.github/workflows/code-quality.yml` workflow runs **on every PR and push to main**:

1. **Sets up Python 3.11** environment
2. **Installs dependencies** via Poetry
3. **Checks Black formatting** (fails if code is not formatted)
4. **Checks Ruff linting** (fails if linting issues exist)
5. **Runs full test suite** with pytest
6. **Uploads coverage** to Codecov
7. **Reports results** as PR status checks

### When It Runs

- ✅ On every **pull request** targeting `main`
- ✅ On every **push** to `main`

### Viewing Results

**On a Pull Request:**
- GitHub shows workflow status at the bottom
- Look for "Checks" section
- Shows pass/fail for each step
- Click to see detailed logs

**On Main Branch:**
- Check the "Actions" tab in GitHub
- See historical workflow runs
- Inspect logs for debugging

## Workflow: Local to Remote

### Typical Development Cycle

1. **Create feature branch**
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make changes**
   ```bash
   # Edit src/sage/my_feature.py
   # Edit tests/test_my_feature.py
   ```

3. **Stage and commit** (hook runs automatically)
   ```bash
   git add .
   git commit -m "feat: add my feature"
   # ↓ Pre-commit hook runs automatically
   # ↓ Code formatted by Black
   # ↓ Issues fixed by Ruff
   # ↓ Changes re-staged
   # ✓ Commit succeeds
   ```

4. **Push to GitHub**
   ```bash
   git push -u origin feat/my-feature
   ```

5. **GitHub Actions runs** (workflow starts automatically)
   - Checks formatting (should pass since hook already fixed it)
   - Checks linting (should pass since hook already fixed it)
   - Runs all tests
   - Uploads coverage

6. **Open Pull Request** on GitHub

7. **GitHub shows results** as PR status checks
   - ✓ All checks must pass before merging
   - ✓ Code is clean, formatted, linted, and tested

## Manual Code Quality Checks

You can run these manually anytime:

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Auto-fix linting issues
poetry run ruff check . --fix

# Run tests
poetry run pytest -v

# Run tests with coverage
poetry run pytest --cov=src --cov-report=html
```

## Configuration

### Pre-Commit Hook

- **Location:** `.git/hooks/pre-commit`
- **Made executable:** `chmod +x .git/hooks/pre-commit`
- **Languages checked:** Only Python (`.py`)
- **Tools used:** Black, Ruff
- **Re-stages changes:** Yes

### GitHub Actions Workflow

- **Location:** `.github/workflows/code-quality.yml`
- **Triggers:** PR to main, push to main
- **Python version:** 3.11
- **Coverage reporting:** Codecov
- **Cache:** Poetry virtualenv

### Tool Configuration

All tools are configured in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov-report=term-missing --cov-report=html"
```

## Troubleshooting

### Pre-commit hook won't run

```bash
# Check if it's executable
ls -la .git/hooks/pre-commit

# Make it executable if needed
chmod +x .git/hooks/pre-commit
```

### Pre-commit hook fails

```bash
# Run tools manually to see what's wrong
poetry run black .
poetry run ruff check . --fix

# Then try committing again
git add .
git commit -m "your message"
```

### GitHub Actions failing

1. Go to repo → "Actions" tab
2. Click the failing workflow run
3. Expand the failed step to see the error
4. Common issues:
   - Code not formatted (run `poetry run black .`)
   - Linting errors (run `poetry run ruff check . --fix`)
   - Tests failing (run `poetry run pytest -v`)

### Poetry cache issues on Actions

The workflow caches the virtualenv. If dependencies change but cache isn't invalidated:

```bash
# Local: Update lock file
poetry lock

# Push this change and Actions will detect new lock file hash
git add poetry.lock
git commit -m "chore: update dependencies"
git push
```

## Benefits

✅ **Consistency** - All code follows same style and standards
✅ **Automation** - No manual formatting/linting needed
✅ **Prevention** - Bad code never reaches main branch
✅ **Confidence** - PRs are tested before merge
✅ **Visibility** - Clear status checks on all PRs
✅ **Coverage** - Test coverage tracked over time

---

**Result:** Every merged PR is automatically formatted, linted, tested, and documented.
