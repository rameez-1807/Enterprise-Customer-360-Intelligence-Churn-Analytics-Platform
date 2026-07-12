# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-12

### Added
- **CI/CD Automation Pipeline**: Integrated GitHub Actions CI config (`.github/workflows/ci.yml`) supporting multi-version Python environments, auto-testing, and security scans.
- **Docker Containerization**: Designed standard `Dockerfile` using multi-stage builds and a `docker-compose.yml` local stack. Added a `Makefile` for developer lifecycle commands.
- **GitHub Community Guidelines**: Configured standard templates including `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `CODEOWNERS`, `.editorconfig`, and `.pre-commit-config.yaml`.
- **Streamlit Glassmorphic Design System**: Introduced centralized UI widgets inside `src/dashboard/design_system.py` and modular card rendering wrappers in `src/dashboard/components.py`.
- **Comprehensive Unit Testing Suite**: Developed unit tests under `tests/unit/` covering data cleaning modules, model configurations, explainability components, and the cached singleton dashboard facade.
- **Cloud Deployment Guides**: Added dedicated deployment documentation in `docs/deployment.md` covering AWS ECS, Azure, GCP Cloud Run, and Streamlit Cloud.
- **Author Credits**: Inserted interactive sidebar footer credits designating the Principal Engineer: `Created by Md Rameez Ahmad`.

### Changed
- **Facade Data Cache**: Transformed the Streamlit app to utilize a centralized facade layer (`DashboardDataService`) to prevent redundant data processing runs during page transitions.
- **View Refactoring**: Refactored `executive_view.py` and `about.py` layout to conform to the responsive glassmorphism UI/UX patterns.
- **Improved Code Quality Gate**: Standardized formatting using `black` and `isort`, resolved Mypy static analysis warnings, and aligned import structures.

### Fixed
- **Blank Streamlit View Bug**: Resolved auto-routing errors by renaming directory `src/dashboard/pages/` to `src/dashboard/views/` to bypass Streamlit's default page loader.
- **Variable Typing Issues**: Resolved undefined variables like `grades_map` in the health engine reporting loops.
