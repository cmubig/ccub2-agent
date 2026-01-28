# Contributing to CCUB2-Agent

Thank you for your interest in contributing to CCUB2-Agent! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Getting Started

CCUB2-Agent is a self-improving, model-agnostic cultural bias mitigation system for generative image models. Before contributing, please:

1. Read the [README.md](README.md) to understand the project
2. Check existing [issues](https://github.com/cmubig/ccub2-agent/issues) and [pull requests](https://github.com/cmubig/ccub2-agent/pulls)
3. Join our community discussions (if applicable)

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (12GB+ VRAM recommended)
- Conda or virtualenv
- Node.js 18+ (for GUI development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cmubig/ccub2-agent.git
   cd ccub2-agent
   ```

2. **Create conda environment**
   ```bash
   conda create -n ccub2 python=3.10
   conda activate ccub2
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Firebase credentials**
   - Obtain Firebase service account JSON from project admin
   - Place it as `firebase-service-account.json` in project root
   - Or set `FIREBASE_SERVICE_ACCOUNT` environment variable

5. **Initialize data**
   ```bash
   python scripts/01_setup/init_dataset.py --country korea
   python scripts/data_processing/enhance_captions.py --country korea
   python scripts/indexing/build_clip_image_index.py --country korea
   python scripts/indexing/integrate_knowledge_to_rag.py --country korea
   ```

### GUI Development

```bash
cd gui/backend
python main.py  # Backend on :8000

# In separate terminal:
cd gui/frontend
npm install
npm run dev  # Frontend on :3000
```

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names (avoid single letters except loop counters)

**Formatting tools:**
```bash
# Format code
black ccub2_agent/ scripts/

# Check linting
flake8 ccub2_agent/ scripts/

# Type checking
mypy ccub2_agent/
```

### Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings:
  ```python
  def function_name(param1: str, param2: int) -> bool:
      """Brief description of function.

      More detailed description if needed.

      Args:
          param1: Description of param1
          param2: Description of param2

      Returns:
          Description of return value

      Raises:
          ValueError: When param2 is negative
      """
  ```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add support for German language`
- `fix: correct CLIP index loading bug`
- `docs: update installation guide`
- `refactor: simplify VLM scoring logic`
- `test: add integration tests for pipeline`
- `chore: update dependencies`

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/integration/test_full_pipeline.py

# Run with coverage
pytest --cov=ccub2_agent --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Integration tests go in `tests/integration/`
- Unit tests go in `tests/unit/`
- Use descriptive test names: `test_vlm_detector_returns_valid_scores()`
- Mock external dependencies (Firebase, HuggingFace Hub)

**Example test:**
```python
import pytest
from ccub2_agent.modules.vlm_detector import VLMCulturalDetector

def test_vlm_detector_initialization():
    """Test VLM detector initializes correctly."""
    detector = VLMCulturalDetector(
        vlm_model="Qwen/Qwen3-VL-8B-Instruct",
        index_dir="data/cultural_index/korea",
        clip_index_dir="data/clip_index/korea",
    )
    assert detector is not None
    assert detector.vlm is not None
```

## Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run tests**
   ```bash
   pytest
   ```

3. **Format code**
   ```bash
   black .
   flake8 .
   ```

4. **Update documentation**
   - Update README.md if adding features
   - Add docstrings to new functions
   - Update CHANGELOG.md

### Submitting PR

1. **Create descriptive title**
   - Good: `feat: add support for Italian language indices`
   - Bad: `Update files`

2. **Write detailed description**
   ```markdown
   ## Summary
   Brief description of changes

   ## Changes
   - Added Italian language support
   - Updated CLIP indexing for multi-language
   - Added tests for Italian data

   ## Testing
   - Ran full pipeline with Italian dataset
   - All tests passing

   ## Screenshots (if applicable)
   ![Screenshot](url)
   ```

3. **Link related issues**
   - Use keywords: `Fixes #123` or `Relates to #456`

4. **Request review**
   - Tag relevant maintainers
   - Be responsive to feedback

### Review Process

- At least one approval required
- All CI checks must pass
- No merge conflicts
- Documentation updated
- Tests added for new features

## Reporting Issues

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, GPU model
- **Steps to reproduce**: Minimal code to reproduce bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Error messages, stack traces
- **Screenshots**: If applicable

### Feature Requests

Use the feature request template and include:

- **Problem**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives**: Other solutions considered
- **Additional context**: Examples, mockups, etc.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the issue, not the person
- Help others learn and grow

## Questions?

- Open a [discussion](https://github.com/cmubig/ccub2-agent/discussions)
- Check existing [documentation](docs/)
- Contact maintainers at chans@andrew.cmu.edu

---

**Thank you for contributing to CCUB2-Agent!** 🎉
