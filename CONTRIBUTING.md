# Contributing to NeuroShard

Thank you for your interest in contributing to NeuroShard! We welcome contributions from the community.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 3. Run Tests

```bash
pytest tests/
```

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in [Issues](https://github.com/neuroshard-ai/neuroshard/issues)
- If not, create a new issue with:
  - Clear title and description
  - Steps to reproduce
  - Expected vs actual behavior
  - System information (OS, Python version, GPU)

### Suggesting Features

- Open an issue with the `enhancement` label
- Describe the feature and its use case
- Explain why it would benefit NeuroShard

### Pull Requests

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style

3. **Write tests** for new functionality

4. **Run tests** to ensure nothing breaks:
   ```bash
   pytest tests/
   ```

5. **Commit** with a clear message:
   ```bash
   git commit -m "Add: brief description of changes"
   ```

6. **Push** and create a Pull Request

### Commit Message Format

```
Type: Brief description

- Add: New feature
- Fix: Bug fix
- Update: Enhancement to existing feature
- Refactor: Code restructuring
- Docs: Documentation only
- Test: Adding tests
```

## Code Style

- **Python**: Follow PEP 8, use Black for formatting
- **Line length**: 100 characters max
- **Type hints**: Use them for function signatures
- **Docstrings**: Use Google style

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description of the function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    pass
```

## Areas for Contribution

### High Priority
- [ ] Performance optimizations
- [ ] Additional aggregation strategies
- [ ] Improved error handling
- [ ] Documentation improvements

### Good First Issues
- Documentation fixes
- Test coverage improvements
- Minor bug fixes
- Code cleanup

## Questions?

- **Discord**: [discord.gg/4R49xpj7vn](https://discord.gg/4R49xpj7vn)
- **Twitter**: [@shardneuro](https://x.com/shardneuro)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
