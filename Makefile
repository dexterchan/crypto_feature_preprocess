.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

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

lint/flake8: ## check style with flake8
	flake8 crypto_feature_preprocess tests

lint: lint/flake8 ## check style



test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source crypto_feature_preprocess setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/crypto_feature_preprocess.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ crypto_feature_preprocess
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

setup_env:
	mamba install -c conda-forge -y --file requirements.txt
	mamba install -c conda-forge -y --file requirements_jupyter.txt
	mamba install -c conda-forge -y --file requirements_dev.txt

test: export DATA_DIR=$(shell pwd)/../crypto_market_data/data
test:
	pytest tests

build_conda:
	sh conda/build_conda.sh


bump_version_patch:
	bump2version patch --allow-dirty

bump_version_minor:
	bump2version minor --allow-dirty 

bump_version_major:
	bump2version major --allow-dirty 

exec_cmd: export DATA_DIR=$(shell pwd)/../crypto_market_data/data
exec_cmd:
	python3 -m crypto_feature_preprocess --exchange kraken --symbol ETHUSD --input_data_dir=${DATA_DIR} --output_data_dir=/tmp/output \
	--start_date 20200101  --time_windows 1095 --data_length 3 --data_step 1 --split_ratio 0.8 --candle_size 15

run_jupyter: export PYTHONPATH=$(shell pwd):$PYTHONPATH
run_jupyter:
	cd notebooks && jupyter lab