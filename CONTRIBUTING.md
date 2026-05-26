# Contributing to BCI Imagined Speech Decoding

First off, thank you for considering contributing to this project! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## How Can I Contribute?

### Reporting Bugs
If you find a bug, please create an issue using the **Bug Report** template. Include:
- A clear description of the issue.
- Steps to reproduce the bug.
- Any relevant logs or screenshots.

### Suggesting Enhancements
We welcome ideas for new features or improvements. Please use the **Feature Request** template for this.

### Pull Requests
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Run tests to ensure everything is working: `make test`.
5. Commit your changes (`git commit -am 'Add some feature'`).
6. Push to the branch (`git push origin feature/your-feature`).
7. Open a Pull Request.

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/charansaiponnad/EdgeRouterNet-Imagined-Speech.git
   cd EdgeRouterNet-Imagined-Speech
   ```
2. Set up the environment:
   ```bash
   make setup
   ```
3. Generate mock data for testing:
   ```bash
   make data-mock
   ```

## Coding Standards
- Follow PEP 8 for Python code.
- Ensure all new features are accompanied by appropriate tests.
- Keep documentation up to date.

## Code of Conduct
Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
