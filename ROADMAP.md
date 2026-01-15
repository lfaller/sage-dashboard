# SAGE Dashboard Development Roadmap

## Current Status: Phase 5 Complete ✅

**What's Done:**
- ✅ MVP: Poetry setup, CI/CD pipeline, core metrics module
- ✅ Phase 1A: Supabase integration, study database functions
- ✅ Phase 1B: Study Search page with advanced filtering & pagination
- ✅ Phase 2: Disease Explorer with disease mapping & drill-down
- ✅ Phase 3: Sex inference system with Rescue Finder page
- ✅ Phase 4A: Real GEO data integration with GEOparse
- ✅ Phase 5: Weekly progress tracking & Trends page with automation
- ✅ 129 tests passing (89% coverage)
- ✅ Automated code quality (Black + Ruff)
- ✅ CHANGELOG.md tracking
- ✅ Centralized logging configuration
- ✅ Streamlit Cloud deployment (live at https://sage-dashboard.streamlit.app)

**Next Steps:** Phase 4B - Automated Entrez API querying, Phase 6 - Polish & Launch

---

## Phase 1A: Database Integration ✅ Complete

### Goal
Connect to Supabase and load real (small sample) study data

### Tasks

#### 1. Set Up Supabase Project
- [ ] Create Supabase account
- [ ] Create new project
- [ ] Get API credentials (URL + key)
- [ ] Store in `.streamlit/secrets.toml` (local only)

**Time:** ~15 min

#### 2. Create Database Schema
- [ ] Run SQL migrations to create tables:
  - `studies` - Study metadata
  - `samples` - Sample-level data
  - `disease_mappings` - Disease term mappings
  - `completeness_snapshots` - Historical tracking
- [ ] Add indexes for common queries
- [ ] Enable Row Level Security (RLS) policies

**Time:** ~30 min

**Files to create/modify:**
- `src/sage/database.py` - Supabase connection utilities
- `sql/schema.sql` - Database schema

#### 3. Implement Database Module (TDD)
- [ ] Write tests first: `tests/test_database.py`
  - Test Supabase connection
  - Test query builders
  - Test error handling
- [ ] Implement `src/sage/database.py`
  - `get_supabase_client()`
  - `fetch_overview_stats()`
  - `search_studies(filters)`

**Time:** ~1 hour

**Test cases:**
```python
def test_get_supabase_client_initializes()
def test_fetch_overview_stats_returns_dict()
def test_search_studies_with_organism_filter()
def test_search_studies_with_limit()
def test_database_connection_error_handling()
```

#### 4. Load Sample Data
- [ ] Find 100-200 real studies from GEO
- [ ] Parse and format data
- [ ] Insert into Supabase
- [ ] Verify with manual query

**Time:** ~30 min

**Data source:**
- GEO API (https://www.ncbi.nlm.nih.gov/geo/info/developer/)
- Or use sample data from `local/` if available

#### 5. Update Overview Page with Real Data
- [ ] Replace mock data with Supabase queries
- [ ] Add caching with `@st.cache_data`
- [ ] Update metrics calculations
- [ ] Test with real data

**Time:** ~30 min

**Files to update:**
- `pages/1_📊_Overview.py`
- Use `calculate_completeness_percentage()` with real numbers

---

## Phase 1B: Study Search Page ✅ Complete

### Goal
Enable users to search and filter studies (COMPLETED)

### Tasks

#### 1. Create Search Module (TDD)
- [ ] Write tests: `tests/test_search.py`
- [ ] Implement `src/sage/search.py`
  - Filter builder
  - Query executor
  - Result formatter

**Time:** ~1 hour

#### 2. Build Search Page
- [ ] Create `pages/2_🔍_Study_Search.py`
- [ ] Sidebar filters:
  - Organism (dropdown)
  - Platform (dropdown)
  - Sex metadata status (missing/partial/complete)
  - Sample size range (slider)
- [ ] Results table with:
  - Study ID (clickable?)
  - Title
  - Organism
  - Sample count
  - Sex metadata %

**Time:** ~1 hour

#### 3. Add Pagination/Export
- [ ] Paginate results (100 per page)
- [ ] Export to CSV button
- [ ] Show result count

**Time:** ~30 min

---

## Phase 2: Disease Mapping ✅ Complete

### Goal
Categorize studies by disease and identify high-value opportunities (COMPLETED)

### Tasks

#### 1. NLP-based Disease Extraction
- [ ] Add `textblob` or `spacy` to dependencies
- [ ] Create `src/sage/disease_extraction.py`
- [ ] Extract disease terms from study titles/summaries
- [ ] Map to Disease Ontology (DOID)

**Time:** ~2 hours

#### 2. Disease Explorer Page
- [ ] Create `pages/3_🔬_Disease_Explorer.py`
- [ ] Show diseases with:
  - Study count
  - Sex metadata completeness %
  - Known sex differences (flag)
  - Clinical priority score
- [ ] Drill-down to studies per disease

**Time:** ~1.5 hours

---

## Phase 3: Sex Inference (After 2)

### Goal
Identify studies where sex can be inferred from expression data

### Tasks

#### 1. Implement Sex Inference Algorithm
- [ ] Create `src/sage/sex_inference.py`
- [ ] Logic based on:
  - XIST (female marker)
  - Y-linked genes (male markers)
  - Coverage patterns
- [ ] Write comprehensive tests

**Time:** ~2 hours

#### 2. Rescue Opportunity Finder Page
- [ ] Create `pages/4_🎯_Rescue_Finder.py`
- [ ] Score high-value reanalysis opportunities:
  - Missing sex metadata ✓
  - RNA-seq (inferrable) ✓
  - Large sample size (>50) ✓
  - Known disease sex differences ✓
- [ ] Rank by ROI score
- [ ] Export reanalysis protocol

**Time:** ~1.5 hours

---

## Phase 4A: Real GEO Data Integration ✅ Complete

### Goal
Load real genomic study data from NCBI GEO using GEOparse (replaced synthetic demo data)

### What Was Done
- ✅ Added GEOparse dependency (^2.0.4)
- ✅ Created `src/sage/geo_fetcher.py` with RateLimiter and NCBI compliance
- ✅ Implemented sex metadata detection from sample names (study-level)
- ✅ Created `scripts/fetch_geo_studies.py` CLI with:
  - 55+ curated human RNA-seq studies from NCBI GEO
  - `--dry-run` for testing without database writes
  - `--skip-existing` for resumable/incremental fetches
  - Configurable rate limiting (default 2 req/sec)
  - Automatic sex inferrability calculation
- ✅ Comprehensive test suite: 25 new tests for geo_fetcher
- ✅ 100% test coverage for core rate limiting and fetching logic
- ✅ Integration with existing Study dataclass and sex inference pipeline
- ✅ Proper logging with centralized logger (no print statements)

### How to Use
```bash
# Test with 5 studies (no database writes)
poetry run python scripts/fetch_geo_studies.py --dry-run --limit 5

# Fetch 50 real studies into database
poetry run python scripts/fetch_geo_studies.py --limit 50

# Resume with skip-existing (fetch only new studies)
poetry run python scripts/fetch_geo_studies.py --skip-existing --limit 200

# High rate limit (requires NCBI API key)
poetry run python scripts/fetch_geo_studies.py --rate-limit 8.0
```

### Key Features
- **Rate Limiting**: NCBI-compliant (default 2 req/sec, configurable up to 10)
- **Resilience**: Retries failed studies with exponential backoff
- **Progress Tracking**: Shows real-time progress and summary statistics
- **Sex Metadata Detection**: Analyzes sample names for M/F patterns
- **Dry-Run Support**: Test without modifying database
- **Resumable**: Skip already-fetched studies with `--skip-existing`

### MVP Design Decisions
- **Curated Accession List**: Start with known human RNA-seq studies (Phase 4B will add automated Entrez querying)
- **Study-Level Only**: No full sample data download (Phase 4B enhancement)
- **GEOparse Library**: Purpose-built for GEO, handles parsing automatically

---

## Phase 4B: Automated Entrez API Querying (Future)

### Goal
Replace curated accession list with automated NCBI Entrez queries

### Planned Tasks
- [ ] Use Biopython Entrez module for automated GEO searches
- [ ] Query: human organism, RNA-seq type, date range (2021-2026)
- [ ] Paginate through results automatically
- [ ] Support incremental updates (fetch only new studies since last run)
- [ ] Track query metadata for reproducibility

**Time:** ~2 hours

---

## Phase 5: Progress Tracking (After 4A)

### Goal
Historical snapshots to measure community improvement

### Tasks

#### 1. Snapshot System
- [ ] Create `scripts/create_snapshot.py`
- [ ] Captures monthly metrics:
  - Total studies
  - With sex metadata %
  - Inferrable count
  - Analyzed by sex %
- [ ] Store in `completeness_snapshots` table

**Time:** ~1 hour

#### 2. Trends Page
- [ ] Create `pages/5_📈_Trends.py`
- [ ] Line charts showing:
  - Sex metadata completeness over time
  - Studies with analysis over time
  - By organism, disease category
- [ ] Year-over-year comparison

**Time:** ~1.5 hours

#### 3. GitHub Actions Scheduled Update
- [ ] Modify `.github/workflows/` for weekly snapshots
- [ ] Automated data refresh

**Time:** ~1 hour

---

## Phase 5: Polish & Launch (Final)

### Tasks

- [ ] Add authentication (optional)
- [ ] API endpoint for programmatic access
- [ ] Blog post announcement
- [ ] Social media presence
- [ ] Submit to bioinformatics communities
- [ ] Deploy to Streamlit Cloud

---

## Estimated Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| MVP | ✅ Done | |
| 1A: Database | ~3 hours | Ready to start |
| 1B: Search | ~2.5 hours | After 1A |
| 2: Disease | ~3.5 hours | After 1B |
| 3: Inference | ~3.5 hours | After 2 |
| 4: Trends | ~3.5 hours | After 3 |
| 5: Polish | ~2-4 hours | After 4 |

**Total:** ~22-24 hours of focused development

---

## Getting Started: Phase 1A

### Step 1: Create Supabase Project
```bash
# Go to https://supabase.com/dashboard
# Create new project, wait for provisioning
# Go to Settings → API, copy:
#   - Project URL
#   - anon/public key
```

### Step 2: Add Secrets (local only)
```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
[connections.supabase]
SUPABASE_URL = "your-url-here"
SUPABASE_KEY = "your-key-here"
EOF
```

### Step 3: Create Feature Branch
```bash
git checkout -b feat/database-integration
```

### Step 4: Write Tests First
```bash
# Add tests to tests/test_database.py
poetry run pytest -v
# Watch them fail (TDD)
```

### Step 5: Implement Module
```bash
# Create src/sage/database.py
# Implement functions to make tests pass
poetry run pytest -v
# All green!
```

### Step 6: Update Overview Page
```bash
# Replace mock data with real queries
poetry run streamlit run app.py
# Test locally
```

### Step 7: Commit & PR
```bash
git add .
git commit -m "feat: add Supabase database integration"
git push -u origin feat/database-integration
# Open PR on GitHub
# Pre-commit hook auto-formats/lints
# GitHub Actions runs tests
```

---

## Best Practices to Follow

- ✅ Write tests first (TDD)
- ✅ One feature per branch
- ✅ Logical commit grouping
- ✅ Run `poetry run pytest` before pushing
- ✅ Let pre-commit hooks handle formatting
- ✅ Keep PRs focused and reviewable
- ✅ Document as you go

---

## Questions?

- See **CONTRIBUTING.md** for development workflow
- See **AUTOMATION.md** for CI/CD details
- See **DEVELOPMENT.md** for full setup guide
