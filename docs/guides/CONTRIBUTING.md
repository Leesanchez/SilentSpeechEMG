# Contributing to Silent Speech EMG Interface

Thank you for your interest in contributing to the Silent Speech EMG Interface project! This document provides guidelines and instructions for contributing.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Pull Request Process](#pull-request-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/SilentSpeechEMG.git
   cd SilentSpeechEMG
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our [coding standards](#coding-standards)

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Update documentation as needed

5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the documentation with any new features or changes
3. Ensure all tests pass
4. Submit a pull request with a clear description of the changes

### PR Title Format
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Adding tests
- chore: Maintenance tasks

## Coding Standards

We follow PEP 8 with some modifications:

1. Line length: 88 characters (Black formatter default)
2. Use type hints for function arguments and return values
3. Document classes and functions using Google-style docstrings
4. Use meaningful variable names

Example:
```python
def process_emg_signal(
    signal: np.ndarray,
    sampling_rate: int = 1000
) -> np.ndarray:
    """
    Process EMG signal with noise reduction and filtering.
    
    Args:
        signal: Raw EMG signal array
        sampling_rate: Signal sampling rate in Hz
        
    Returns:
        Processed EMG signal array
    """
    # Implementation
```

### Code Formatting
We use the following tools:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run formatting:
```bash
black .
isort .
flake8 .
mypy .
```

## Testing Guidelines

1. Write tests for all new features
2. Maintain test coverage above 80%
3. Use pytest fixtures for common test setups
4. Name test files with `test_` prefix
5. Use descriptive test names that explain the test case

Example:
```python
def test_emg_signal_processing_removes_noise():
    # Test implementation
```

## Documentation

1. Update docstrings for all new functions and classes
2. Keep README.md up to date
3. Add examples for new features
4. Document any breaking changes

### Documentation Structure
```
docs/
├── api/           # API documentation
├── examples/      # Usage examples
├── guides/        # User guides
└── images/        # Documentation images
```

## Electrode Reduction Study

When contributing to the electrode reduction study:

1. Follow the methodology outlined in `docs/guides/electrode_reduction_study.md`
2. Document all assumptions and limitations
3. Include visualizations of results
4. Validate findings with synthetic data before real data
5. Consider impact on system performance

## Questions and Support

- Open an issue for bugs or feature requests
- Join our community discussions
- Contact maintainers for security issues

Thank you for contributing to making silent speech recognition more accessible! 