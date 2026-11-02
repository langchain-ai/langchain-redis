.PHONY: all test tests integration_test integration_tests lint lint_diff lint_package lint_tests format format_diff spell_check spell_fix check_imports help

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

test tests:
	poetry run pytest $(TEST_FILE)

integration_test integration_tests:
	poetry run pytest $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/redis --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_redis
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	poetry run ruff check $(PYTHON_FILES)
	poetry run ruff format $(PYTHON_FILES) --diff
	poetry run ruff check $(PYTHON_FILES) --select I
	mkdir -p $(MYPY_CACHE); poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)
	poetry check

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff check $(PYTHON_FILES) --select I --fix

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_redis -name '*.py')
	poetry run python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports                - check imports'
	@echo 'format                       - run code formatters'
	@echo 'format_diff                  - run code formatters on changed files'
	@echo 'lint                         - run linters'
	@echo 'lint_diff                    - run linters on changed files'
	@echo 'lint_package                 - run linters on langchain_redis'
	@echo 'lint_tests                   - run linters on tests'
	@echo 'spell_check                  - check spelling'
	@echo 'spell_fix                    - fix spelling'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_tests            - run integration tests'
