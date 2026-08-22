.PHONY: configure build build_frontend build_backend install run python_env models clean help unit_tests integration_tests deploy_production

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Setup & Environment:"
	@echo "    make install         Install system dependencies"
	@echo "    make python_env      Create Python virtual env and install deps"
	@echo "    make clean           Remove build artifacts, venv, and caches"
	@echo ""
	@echo "  Development Build:"
	@echo "    make configure       Configure CMake build system"
	@echo "    make models          Generate C++ models from Python scripts"
	@echo "    make build_backend   Build backend C++ code"
	@echo "    make build_frontend  Build frontend React application"
	@echo "    make build           Full build (configure + backend + frontend)"
	@echo ""
	@echo "  Testing:"
	@echo "    make unit_tests      Run unit tests"
	@echo "    make integration_tests  Run integration tests"
	@echo ""
	@echo "  Deployment:"
	@echo "    make restart         Restart Docker containers"
	@echo "    make up         		Up Docker containers"
	@echo "    make down         	Down Docker containers"
	@echo ""
	@echo "  Other:"
	@echo "    make help            Show this help message"

MODELS_DIR := contrib/emotiefflib/models
CONFIG_DIR := config
NGINX_SOURCE := $(CONFIG_DIR)/nginx
NGINX_TARGET := /etc/nginx/sites-available/emotion-ai
SERVICE_SOURCE := $(CONFIG_DIR)/service
SERVICE_TARGET := /etc/systemd/system/emotion-ai.service

install:
	bash install_deps.sh

configure:
	mkdir -p build && cd build && cmake .. -G Ninja

build_backend:
	cd build && ninja -j4

build_frontend:
	cd frontend && npm install && npm run build

build: configure build_backend build_frontend

up:
	docker compose up -d

up-build:
	docker compose up -d --build

down:
	docker compose down

restart: down up

unit_tests:
	cd build && ./tests/EmotionAI_UnitTests

integration_tests:
	cd build && ./tests/EmotionAI_IntegrationTests

python_env: venv
	. venv/bin/activate && pip install -r requirements.txt

venv:
	python3 -m venv venv

models: python_env
	. venv/bin/activate && cd venv && python3 prepare_models_for_emotieffcpplib.py

clean-pyc:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean:
	rm -rf \
		build \
		frontend/node_modules \
		frontend/build \
		$(VENV_DIR) \
		package-lock.json