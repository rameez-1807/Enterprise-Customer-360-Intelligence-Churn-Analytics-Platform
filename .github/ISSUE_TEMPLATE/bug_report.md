name: 🐛 Bug Report
description: Report a bug or issue in the Customer 360 Platform.
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thank you for reporting a bug! Please fill out the details below to help us reproduce and resolve the issue.
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear and concise description of what the bug is.
      placeholder: Describe the bug here...
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Describe the exact steps needed to reproduce the behavior.
      placeholder: |
        1. Ingest raw data using 'main.py'
        2. Launch Streamlit server
        3. Navigate to Customer 360 page
        4. See error...
    validations:
      required: true
  - type: textarea
    id: expected_behavior
    attributes:
      label: Expected Behavior
      description: A clear description of what you expected to happen.
      placeholder: What should have happened...
    validations:
      required: true
  - type: textarea
    id: screenshots
    attributes:
      label: Screenshots & Visual Logs
      description: If applicable, add screenshots or console tracebacks to help explain your problem.
  - type: dropdown
    id: os
    attributes:
      label: Operating System
      options:
        - Windows
        - macOS
        - Linux
        - Other
    validations:
      required: true
  - type: input
    id: python_version
    attributes:
      label: Python Version
      placeholder: e.g., 3.11.5
    validations:
      required: true
