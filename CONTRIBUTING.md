# Contributing Guidelines

Thank you for considering contributing to this project! This document outlines the process for contributing.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template if available
3. Include:
   - Clear description of the issue
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Suggesting Features

1. Open an issue with the "enhancement" label
2. Describe the feature and its use case
3. Explain why it would be valuable

### Submitting Code

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Write or update tests** as needed
6. **Run tests** to ensure nothing is broken
7. **Commit** with a clear message:
   ```bash
   git commit -m "Add: brief description of change"
   ```
8. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Run tests
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines for Python
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise

## Commit Messages

Use clear, descriptive commit messages:

- `Add: new feature or file`
- `Fix: bug fix description`
- `Update: changes to existing feature`
- `Remove: deleted feature or file`
- `Docs: documentation changes`
- `Test: test additions or changes`
- `Refactor: code refactoring`

## Pull Request Guidelines

- Keep PRs focused on a single change
- Update documentation if needed
- Add tests for new functionality
- Ensure all tests pass
- Reference related issues in the PR description

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing!
