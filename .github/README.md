# Continuous Integration (CI)

This project uses GitHub Actions for continuous integration to automatically run tests on every commit and pull request.

## Workflows

### CI Workflow (`.github/workflows/ci.yml`)

The CI workflow runs on every push to `main` and `develop` branches, as well as on pull requests targeting these branches.

**Test Job:**
- Runs on Ubuntu with Python 3.11
- Uses Poetry for dependency management
- Installs test dependencies (`pytest`, `pytest-cov`)
- Runs comprehensive test suite with coverage reporting
- Uploads coverage reports to Codecov

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

### Configuration

- **pytest**: Configured in `pyproject.toml` with test discovery settings
- **coverage**: Reports uploaded to Codecov automatically via CI

## Status Badges

Once the CI is running, you can add status badges to your main README:

```markdown
![CI Status](https://github.com/username/binconvfm/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/username/binconvfm/branch/main/graph/badge.svg)](https://codecov.io/gh/username/binconvfm)
```