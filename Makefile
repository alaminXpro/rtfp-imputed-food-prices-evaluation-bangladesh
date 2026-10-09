.PHONY: data test

PY := .venv/bin/python

data:
	$(PY) -I -m rtfp.download --root data/raw

test:
	$(PY) -m pytest -q
