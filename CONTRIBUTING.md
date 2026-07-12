# Contributing to Enterprise Customer 360 Platform

First of all, thank you for taking the time to contribute! We welcome contributions from everyone and appreciate your efforts to improve this enterprise analytics platform.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs
* Search existing issues to verify the bug has not been reported.
* Open a new issue using the Bug Report template, detailing your environment, steps to reproduce, and actual vs. expected behavior.

### Suggesting Enhancements
* Search existing issues for similar requests.
* Explain the feature's business case and technical implementation details.

### Submitting Pull Requests
1. Fork the repository and create your branch from `staging`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write tests covering your modifications.
3. Verify that all pytest unit tests pass cleanly:
   ```bash
   python -m pytest
   ```
4. Verify that code matches the style guide (Black formatting, PEP8 compliance).
5. Submit your Pull Request using the Pull Request template.

## Coding Style & Standards
* Follow standard PEP 8 guidelines.
* Line length limit: **120 characters**.
* Use explicit type hints for all public functions, classes, and methods.
* Docstrings must conform to Google Style formatting.
* Avoid print statements in production code; use the structured JSON logger.
* Format code using `black` and `isort`.
