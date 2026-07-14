.PHONY: clean clean-test clean-pyc clean-build sync lint test coverage docs servedocs benchmarks examples notebooks release dist install help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

sync: ## create/update the uv-managed virtualenv from uv.lock
	uv sync

lint: ## check style with flake8
	uv run flake8 flatland tests examples benchmarks

test: ## run tests quickly with the default Python
	echo "$$DISPLAY"
	uv run pytest

coverage: ## check code coverage quickly with the default Python
	uv run python make_coverage.py

docs: ## generate Sphinx HTML documentation, including API docs
	uv run python make_docs.py

servedocs: docs ## compile the docs watching for changes
	uv run watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

benchmarks: ## run all benchmarks
	uv run python benchmarks/benchmark_all_examples.py

examples: ## run all examples
	uv run python benchmarks/run_all_examples.py

notebooks: ## run all notebooks
	uv run python notebooks/run_all_notebooks.py

release: dist ## package and upload a release
	uv publish

dist: clean ## builds source and wheel package
	uv build
	ls -l dist

install: clean ## install the package into the uv-managed virtualenv
	uv sync
