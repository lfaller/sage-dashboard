# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Weekly snapshot workflow now installs project package correctly (removed `--no-root` flag)
- Trends view now displays full 5-year historical data (increased snapshot limit from 52 to 260)

### Added
- `pages/6_❓_FAQ.py` - Comprehensive FAQ page documenting Trends methodology, metrics, and snapshot creation
- Phase 4B: Automated NCBI Entrez API querying for dynamic study discovery
- `src/sage/entrez_searcher.py` - EntrezSearcher module for NCBI GEO searches with rolling 5-year window
- Automated study discovery via `--search` flag in `scripts/fetch_geo_studies.py`
- Support for organism and study type filtering in automated searches
- 8 new tests for EntrezSearcher functionality (139 tests total, 88% coverage)

### Changed
- Backfilled 5 years of historical snapshots for complete historical trend visualization
- Updated `scripts/fetch_geo_studies.py` to support both curated list (default) and automated search modes
- Enhanced fetch_geo_studies.py CLI with new arguments: `--search`, `--organism`, `--study-type`, `--years-back`, `--email`
- Updated README.md with automated search examples
- Updated ROADMAP.md to mark Phase 4B complete
- Biopython (^1.84) added to dependencies for Entrez API access

### Technical Details
- EntrezSearcher reuses existing RateLimiter class (NCBI-compliant)
- Converts GDS IDs (from Entrez) to GSE accessions (for GEOparse) automatically
- Rolling 5-year date window updates automatically without manual configuration
- Error handling for partial results: skips failed conversions and continues
- Email address required for NCBI Entrez (defaults to sage-dashboard@example.com)

## [0.5.0] - 2026-01-15

### Added
- Phase 5: Weekly progress tracking system with automated Trends page
- Snapshot system for capturing weekly metrics (`create_snapshot()`, `fetch_snapshots()`)
- `pages/5_📈_Trends.py` - Historical progress visualization with 4 key charts
- `scripts/create_snapshot.py` - CLI for manual snapshot creation with filtering options
- `.github/workflows/weekly-snapshot.yml` - GitHub Actions automation for weekly snapshots (Monday 00:00 UTC)
- Snapshot functions with organism and disease category filtering
- 5 new tests for snapshot functionality (129 tests total, 89% coverage)

### Changed
- Updated ROADMAP.md to mark Phase 5 complete
- Updated README.md with Deployment section for Streamlit Cloud
- Version bumped to 0.5.0 for new feature release
- Improved database.py with 180+ new lines for snapshot infrastructure

### Technical Details
- Snapshots stored in `completeness_snapshots` table (existing schema)
- Weekly automation requires SUPABASE_URL and SUPABASE_KEY GitHub secrets
- Supports creating snapshots for: overall, by organism, by disease category, or all combinations
- 10-minute cache TTL for `fetch_snapshots()` for performance
- GitHub Actions can be triggered manually via `workflow_dispatch` for testing

## [0.4.0] - 2026-01-14

### Added
- Phase 4A: Real GEO data integration with GEOparse library
- GEOFetcher module for fetching metadata from NCBI GEO (src/sage/geo_fetcher.py)
- RateLimiter class for NCBI-compliant API rate limiting (default 2 req/sec)
- Sex inferrability calculation integrated into fetch stage with sample name analysis
- Organism detection with taxid fallback (sample_taxid, platform_taxid fields)
- CLI script for batch downloading GEO studies (scripts/fetch_geo_studies.py)
- Curated list of 55+ human RNA-seq studies from diverse disease areas
- 25 new tests for geo_fetcher module with 89% coverage
- Comprehensive mock fixtures for GEO response testing

### Changed
- Moved sex inferrability calculation from post-fetch to GEO fetch stage
- Enhanced organism metadata extraction to handle missing organism fields
- Improved sex inference accuracy on real data (22% -> 60% inferrability on test set)
- Updated README.md with NCBI GEO data sources section and fetch_geo_studies.py usage examples
- Updated ROADMAP.md to mark Phase 4A complete

### Technical Details
- Integrates with existing Study dataclass and sex inference pipeline
- Supports --dry-run mode for testing without database writes
- Supports --skip-existing flag for resumable/incremental fetching
- Configurable rate limiting for API key users (up to 10 req/sec)
- Exponential backoff retry logic for transient failures
- All 124 tests passing with 92% code coverage

## [0.3.1] - 2026-01-14

### Added
- Centralized logging configuration module (`src/sage/logging_config.py`)

### Changed
- Refactored debug output from print statements to proper Python logging module
- Updated `get_rescue_opportunities()` to use logger instead of stderr printing
- Improved debugging experience with consistent log format and levels

## [0.3.0] - 2026-01-14

### Added
- 🎯 Rescue Finder page for identifying high-value datasets for sex inference
- Sex inference module with strategy pattern for extensibility
- Metadata-based sex inference (MVP) analyzing study characteristics and sample naming patterns
- Regex-based sample name pattern detection (M/F, Male/Female labels)
- Rescue opportunity scoring algorithm (5-factor weighted model)
- 3 new database functions: `get_rescue_opportunities()`, `calculate_rescue_score()`, `fetch_rescue_stats()`
- Batch processing script (`scripts/update_sex_inferrability.py`) for analyzing all studies
- 26 tests for sex inference module with 95% code coverage
- 14 new database tests with comprehensive mock patterns
- Strategy pattern design for future expression-based inference

### Changed
- Updated AGENTS.md to document TDD caveats with Streamlit caching

## [0.2.0] - 2026-01-14

### Added
- 🔬 Disease Explorer page for browsing diseases by sex metadata completeness
- Disease statistics and metrics (total diseases, study mappings, avg completeness)
- Disease filtering by category, minimum study count, and known sex differences
- Drill-down functionality to view studies for specific diseases
- 4 new database functions: `fetch_disease_stats()`, `get_diseases_with_completeness()`, `get_studies_for_disease()`, `get_disease_categories()`

### Fixed
- Replaced deprecated `use_container_width` parameter with `width="stretch"` in Study Search page
- GitHub Actions workflow cache key generation and permissions for auto-formatting

## [0.1.0] - 2026-01-07

### Added
- Initial MVP with Poetry project setup
- Pre-commit hooks (Black + Ruff automation)
- GitHub Actions CI/CD pipeline with automated code quality checks
- Core metrics module with 100% test coverage
- Streamlit dashboard with home page
- 📊 Overview page with study metadata statistics
- 🔍 Study Search page with advanced filtering and pagination
- Comprehensive test suite (55 tests)
- Database integration with Supabase
