# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

The project is structured as a monorepo with the main library at `libs/redis/`. All development commands should be run from `libs/redis/`:

```bash
cd libs/redis
```

## Virtual Environments

Poetry manages dependencies; some users also use it to automatically manage Python virtual environments.

In this repository you may encounter a project virtual environment already created locally. Common locations:
- The repository root
- `libs/redis/env/`

Common directory names:
- `.venv`
- `env`
- `venv`

Recommended workflow:
1) If a virtual environment exists in the repo, activate it first, then run Python or `make` commands:
```bash
source .venv/bin/activate  # or: source libs/redis/env/bin/activate
make test                  # or any other Make target
```

2) If `poetry` is available on your PATH without activating a venv, you can try to use it directly:
```bash
# From libs/redis/
make test
# or explicitly
poetry run pytest tests/unit_tests/test_specific.py
```

3) If you run `poetry` or `make` and see `poetry: command not found`, Poetry is
not on your PATH. Try to activate the project's virtual environment to see if it
already contains Poetry (e.g., `source libs/redis/env/bin/activate`). If it
doesn't, ask the user if you should install it.

Notes:
- Makefile targets call `poetry run ...`. When a venv is activated and contains
  Poetry, `make` will use that Poetry and run inside that venv. When Poetry is
  on PATH globally, it will use its managed venv and you do not need to activate
  one manually.
- Quick checks:
  - `which poetry`
  - `TEST_FILE=tests/unit_tests/test_specific.py make test`

### Testing
- `make test` - Run unit tests
- `make integration_tests` - Run integration tests (requires OPENAI_API_KEY)
- `TEST_FILE=tests/unit_tests/test_specific.py make test` - Run specific test file

### Linting and Formatting
- `make lint` - Run linters (ruff + mypy)
- `make format` - Auto-format code
- `make check_imports` - Validate import structure
- `make spell_check` - Check spelling with codespell

### Dependencies
- `poetry install --with test` - Install unit test dependencies
- `poetry install --with test,test_integration` - Install all test dependencies
- `poetry install --with lint,typing,test,test_integration` - Install all development dependencies

## Architecture

### Core Components

The library provides three main integrations with Redis:

1. **RedisVectorStore** (`vectorstores.py`) - Vector storage and similarity search
2. **RedisCache/RedisSemanticCache** (`cache.py`) - LLM response caching
3. **RedisChatMessageHistory** (`chat_message_history.py`) - Chat message persistence

### Configuration System

All components use the centralized `RedisConfig` class (`config.py`) which provides:
- Multiple initialization patterns (from_kwargs, from_schema, from_yaml, etc.)
- Pydantic-based validation with smart defaults
- Schema management for Redis index structures
- Connection handling (redis_client or redis_url)

Key design patterns:
- Config validates that only one of `index_schema`, `schema_path`, or `metadata_schema` is specified
- `key_prefix` defaults to `index_name` if not provided
- ULID-based default index names for uniqueness

### Vector Store Implementation

`RedisVectorStore` uses RedisVL (Redis Vector Library) underneath:
- Supports FLAT and HNSW indexing algorithms
- Multiple distance metrics (COSINE, L2, IP)
- Metadata filtering via RedisVL FilterExpression
- Custom cosine similarity implementation with optional simsimd acceleration
- Document storage in either hash or JSON format

### Cache Implementation

Two cache types:
- `RedisCache` - Standard key-value caching with TTL
- `RedisSemanticCache` - Uses embeddings for similarity-based cache hits

Both integrate with LangChain's caching interface and support async operations.

### Chat History Implementation

`RedisChatMessageHistory` provides:
- JSON-based message storage with Redis search indexing
- Message type handling (Human, AI, System)
- TTL support for automatic expiration
- Full-text search capabilities via Redis FT module

## Testing Patterns

- Unit tests in `tests/unit_tests/` - no external dependencies
- Integration tests in `tests/integration_tests/` - require Redis and OpenAI
- Uses pytest with async support
- `conftest.py` provides shared fixtures
- Integration tests use testcontainers for Redis instances

## Import Structure

The library follows a clean import pattern:
- Main exports in `__init__.py`
- Version info in `version.py` with both `__version__` and `__lib_name__`
- Core functionality split by concern (config, vectorstores, cache, chat_history)

All major classes (RedisVectorStore, RedisConfig, RedisCache, etc.) should be imported from the top-level package.