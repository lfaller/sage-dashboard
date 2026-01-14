# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
