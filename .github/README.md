# Continuous Integration (CI)

This project uses GitHub Actions for continuous integration to automatically run tests and linting on every commit and pull request.

## Workflows

### CI Workflow (`.github/workflows/ci.yml`)

The CI workflow runs on every push to `main` and `develop` branches, as well as on pull requests targeting these branches.

**Test Job:**
- Runs on Ubuntu with Python 3.11
- Uses Poetry for dependency management
- Installs test dependencies (`pytest`, `pytest-cov`)
- Runs comprehensive test suite with coverage reporting
- Uploads coverage reports to Codecov

**Lint Job:**
- Runs code quality checks with multiple tools:
  - **Black**: Code formatting
  - **isort**: Import sorting
  - **flake8**: Style guide enforcement
  - **mypy**: Type checking (non-blocking)

## Local Development

### Running Tests
```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest tests/ -v

# Run tests with coverage
poetry run pytest tests/ -v --cov=binconvfm --cov-report=term-missing
```

### Code Quality Checks
```bash
# Format code with Black
poetry run black binconvfm/ tests/

# Sort imports with isort
poetry run isort binconvfm/ tests/

# Check style with flake8
poetry run flake8 binconvfm/ tests/

# Type check with mypy
poetry run mypy binconvfm/ --ignore-missing-imports
```

### Configuration

- **Black**: Configured in `pyproject.toml` with 100 character line length
- **isort**: Configured in `pyproject.toml` to work with Black
- **flake8**: Configured in `.flake8` with E203, W503 ignored for Black compatibility
- **pytest**: Configured in `pyproject.toml` with test discovery settings

## Status Badges

Once the CI is running, you can add status badges to your main README:

```markdown
![CI Status](https://github.com/username/binconvfm/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/username/binconvfm/branch/main/graph/badge.svg)](https://codecov.io/gh/username/binconvfm)
```