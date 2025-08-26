.PHONY: install init run test clean

install:
	pip install -e .

init:
	python scripts/initialize.py

run:
	python run.py

run-prod:
	gunicorn --bind 0.0.0.0:8000 wsgi:app

test:
	python -m pytest tests/ -v

clean:
	rm -rf uploads/*
	rm -rf results/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/models/__pycache__
	rm -rf src/routes/__pycache__
	rm -rf src/utils/__pycache__
	find . -name "*.pyc" -delete
