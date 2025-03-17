# Contributing to Document Assistant

Thank you for considering contributing to Document Assistant! We value every contribution, whether it's code, documentation, bug reports, feature requests, or feedback.

## Code of Conduct

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative and constructive
- Focus on what is best for the community
- Show empathy towards other community members

## Ways to Contribute

1. **Code Contributions**
   - Bug fixes
   - Feature implementations
   - Performance improvements
   - Test coverage improvements

2. **Documentation**
   - README improvements
   - Code comments
   - Usage examples
   - Architecture documentation

3. **Testing**
   - Writing unit tests
   - Integration testing
   - Bug reproduction cases
   - Test coverage improvements

4. **Reviews**
   - Code reviews
   - Documentation reviews
   - Design proposals
   - Architecture feedback

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/Document-Assistant.git
cd Document-Assistant
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Set up environment variables:
```bash
# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

## Development Tools

The project uses several development tools:

- **pytest** (>=7.4.4): Testing framework
- **pytest-cov** (>=4.1.0): Code coverage reporting
- **flake8** (>=7.0.0): Code style and quality checker
- **mypy** (>=1.8.0): Static type checking
- **black** (>=24.2.0): Code formatter

## Testing Guidelines

1. **Writing Tests**
   - Write tests for new features
   - Add tests for bug fixes
   - Maintain or improve coverage
   - Use meaningful test names

2. **Running Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_file.py

# Run with verbose output
pytest -v
```

## Code Style

1. **Formatting**
   - Use black for code formatting:
   ```bash
   black .
   ```
   - Run flake8 for style checks:
   ```bash
   flake8 .
   ```

2. **Type Hints**
   - Add type hints to all functions
   - Run mypy for type checking:
   ```bash
   mypy .
   ```

3. **Documentation**
   - Document all public functions and classes
   - Include docstrings with type information
   - Add inline comments for complex logic

## Git Workflow

1. **Branching**
   - Create feature branches from main
   - Use descriptive branch names
   - Keep branches focused and small

2. **Commits**
   - Write clear commit messages
   - Use conventional commits format
   - Keep commits atomic and focused

3. **Pull Requests**
   - Create detailed PR descriptions
   - Reference related issues
   - Include test coverage
   - Update documentation

## Pull Request Process

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test:
```bash
# Format code
black .

# Run linting
flake8 .

# Run type checking
mypy .

# Run tests
pytest
```

3. Commit your changes:
```bash
git add .
git commit -m "feat: add new feature"
```

4. Push to your fork:
```bash
git push origin feature/your-feature-name
```

5. Create a Pull Request:
   - Use a clear title and description
   - Link related issues
   - Include test results
   - Add documentation updates

## Release Process

1. Version bumping
2. Changelog updates
3. Documentation updates
4. Security checks
5. Final testing
6. Release tagging

## License

By contributing to Document Assistant, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details. 