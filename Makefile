.PHONY: all test test-mypy test-pylint

PYTEST_IGNORE =--ignore=glmm_encoder/examples

test:
	python3 setup.py pytest --addopts '$(PYTEST_IGNORE) --ignore glmm_encoder/examples/ --disable-pytest-warnings --docstyle --mypy --pylint  --pylint-rcfile=setup.cfg'

test-docstyle:
	python3 setup.py pytest --addopts '$(PYTEST_IGNORE) --docstyle'

test-mypy:
	python3 setup.py pytest --addopts '$(PYTEST_IGNORE) --mypy -m mypy'

test-pylint:
	python3 setup.py pytest --addopts '$(PYTEST_IGNORE) --pylint -m pylint --pylint-rcfile=setup.cfg'