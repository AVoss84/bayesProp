# Variables
ROOT_FOLDER := "src/bayesAB"
DOCS_DIR := "docs/mkdocs"

# ─── Documentation ─────────────────────────────────────────────

# Serve docs locally for testing
docs-serve:
    cd {{ DOCS_DIR }} && uv run mkdocs serve

# Build docs
docs-build:
    cd {{ DOCS_DIR }} && uv run mkdocs build

# ─── Code Quality ───────────────────────────────────────────────

# Format and auto-fix lint issues
format:
    uv run ruff format {{ ROOT_FOLDER }}
    uv run ruff check {{ ROOT_FOLDER }} --fix

# Run mypy type checking
type-check:
    uv run mypy {{ ROOT_FOLDER }}

# Run strict mypy type checking
type-check-strict:
    uv run mypy {{ ROOT_FOLDER }} --strict

# Format, lint, and type-check
lint: format
    uv run mypy {{ ROOT_FOLDER }}

# Ruff check only (no fix)
ruff-check:
    uv run ruff check {{ ROOT_FOLDER }}

# Ruff check with auto-fix
ruff-fix:
    uv run ruff check {{ ROOT_FOLDER }} --fix

# ─── Pre-commit ────────────────────────────────────────────────

# Run all pre-commit hooks on staged files
pre-commit:
    uv run pre-commit run

# Run all pre-commit hooks on all files
pre-commit-all:
    uv run pre-commit run --all-files

# ─── Testing ───────────────────────────────────────────────────

# Run all tests
test:
    uv run pytest tests/ -v

# Run tests with coverage
test-cov:
    uv run pytest tests/ -v --cov={{ ROOT_FOLDER }} --cov-report=term-missing
