.PHONY: install init run test clean

install:
	pip install -e .

init:
	python scripts/initialize.py

run:
	python run.py

run-prod:
	gunicorn --bind 0.0.0.0:8000 --workers 4 --threads 2 --timeout 120 wsgi:application

test:
	python -m pytest tests/ -v

clean:
	rm -rf upload/*
	rm -rf result/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/models/__pycache__
	rm -rf src/routes/__pycache__
	rm -rf src/utils/__pycache__
	find . -name "*.pyc" -delete
