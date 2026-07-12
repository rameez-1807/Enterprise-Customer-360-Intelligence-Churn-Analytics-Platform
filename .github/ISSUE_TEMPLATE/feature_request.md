name: 🚀 Feature Request
description: Propose a new feature, model improvement, or analytics dashboard component.
labels: ["enhancement", "feature"]
body:
  - type: markdown
    attributes:
      value: |
        Have an idea to improve the Enterprise Customer 360 Platform? Share it with us here!
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear and concise description of what the problem is (e.g., I'm frustrated when...).
      placeholder: Describe the pain point...
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear and concise description of what you want to happen.
      placeholder: Describe the requested feature...
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: A clear description of any alternative solutions or features you've considered.
  - type: textarea
    id: context
    attributes:
      label: Additional Context
      description: Add any other context, designs, mockups, or screenshots about the feature request here.
