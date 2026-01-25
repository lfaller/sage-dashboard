# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-01-25

### Added
- Phase 6A.2: Training data extraction and management for model validation
- `TrainingDataExtractor` class for extracting training samples
  * `fetch_high_confidence_samples()` to filter by confidence threshold
  * `validate_label_consistency()` to detect conflicting sex labels across sources
  * `resolve_conflicting_labels()` with source preference logic (characteristics prioritized)
  * `export_training_fixture()` for JSON fixture creation with metadata
  * `compute_dataset_statistics()` for balance ratio and label distribution analysis
  * `validate_training_set_size()` to warn on undersized datasets (<20 samples)
- `TrainingDataset` dataclass for label management
  * Properties: `male_count`, `female_count`, `total_samples`, `balance_ratio`
  * `stratified_split()` for balanced train/test splitting with stratification
- `TrainingDatasetMetadata` dataclass for fixture metadata
  * Serialization: `to_dict()` and `from_dict()` for fixture versioning
- 19 new tests for training data management (96% coverage)
  * Tests for extraction, label conflict resolution, statistics computation
  * Tests for stratified splitting on balanced and imbalanced datasets
  * Tests for fixture metadata serialization
- `docs/TESTING_GUIDE.md` - Comprehensive testing reference with 7 testing methods

### Technical Details
- Training data extraction uses stratified k-fold approach to handle small datasets
- Preference-based conflict resolution: characteristics > sample_names > other sources
- Stratified splitting maintains class balance ratio in train/test splits
- Fixture versioning supports reproducible model training and validation
- Integration with Phase 6A.1 expression-based inference pipeline
- Overall test coverage: 214 tests passing, 90% coverage
  * training_data_manager.py: 96% coverage

## [0.7.0] - 2026-01-23

### Added
- Phase 6A.1: Expression-based sex inference using elastic net logistic regression
- `ExpressionFetcher` class for fetching gene expression data from GEO samples
  * Flexible column detection for gene symbols and expression values
  * Support for X/Y chromosome sex marker genes (XIST, RPS4Y1, DDX3Y, EIF1AY, KDM5D)
  * Handles platform heterogeneity with fallback column detection
  * Single-sample and batch extraction methods
- `SexClassifier` class for elastic net model training and prediction
  * Train elastic net logistic regression on normalized expression data
  * Probability scoring (P(male) in [0,1] range)
  * Three-way classification: male/female/ambiguous (threshold 0.7)
  * Batch classification via `classify_study()` for entire datasets
- `ElasticNetInferenceStrategy` implementing strategy pattern
  * Replaces placeholder InferenceStrategy implementation
  * End-to-end inference: fetch expression → classify samples → aggregate statistics
  * Graceful degradation on expression fetch failures
  * Returns InferenceResult with sample-level classifications
- 26 new tests for expression-based sex inference (414 test lines)
  * Gene selection, column detection, expression extraction tests
  * Model training, probability scoring, classification tests
  * Batch processing and strategy pattern integration tests
  * All tests use mocking to avoid network dependencies
- scikit-learn ^1.3.0 dependency for elastic net logistic regression

### Technical Details
- Based on Flynn et al. (2021) BMC Bioinformatics methodology
- Elastic net combines L1 (lasso) and L2 (ridge) for robust feature selection across platform heterogeneity
- Default classification threshold 0.7 (P(male) >= 0.7 → male, P(male) <= 0.3 → female, else ambiguous)
- Expression data preprocessed as 1D feature vectors: [XIST, RPS4Y1, DDX3Y, EIF1AY, KDM5D]
- Validation against Flynn et al. benchmarks planned for Phase 6A.2
- Overall test coverage: 195 tests passing, 90% coverage
  * expression_fetcher.py: 93% coverage
  * sex_inference.py: 96% coverage

## [0.6.0] - 2026-01-23

### Fixed
- Weekly snapshot workflow now installs project package correctly (removed `--no-root` flag)
- Trends view now displays full 5-year historical data (increased snapshot limit from 52 to 260)

### Added
- Sex/gender metadata extraction from GEO sample characteristics_ch1 fields (supplement to sample name detection)
- `extract_sex_from_characteristics()` - Parse sex from various characteristic formats (sex:, gender:, sex|, sex=)
- `analyze_sample_characteristics()` - Analyze sex metadata across all samples in study
- `merge_sex_analyses()` - Merge characteristics and sample name detection with priority logic
- Support for multiple sex encoding formats: "sex: male", "gender: F", "sample_sex: Female", "sex=M", etc.
- 30 new test cases for characteristics parsing and merging logic (86 tests total, 95% coverage for sex_inference.py)
- `pages/6_❓_FAQ.py` - Comprehensive FAQ page documenting Trends methodology, metrics, and snapshot creation
- Phase 4B: Automated NCBI Entrez API querying for dynamic study discovery
- `src/sage/entrez_searcher.py` - EntrezSearcher module for NCBI GEO searches with rolling 5-year window
- Automated study discovery via `--search` flag in `scripts/fetch_geo_studies.py`
- Support for organism and study type filtering in automated searches
- 8 new tests for EntrezSearcher functionality (139 tests total, 88% coverage)

### Changed
- `detect_sex_metadata_from_gse()` now analyzes both sample names and characteristics_ch1
- `_calculate_sex_inferrability()` now merges characteristics-based and name-based sex detection
- Characteristics prioritized over sample names as more explicit/reliable source
- Expected improvement: +10-20% increase in sex metadata detection for studies with explicit characteristics
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
