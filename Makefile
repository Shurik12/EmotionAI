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
	@echo "    make deploy_production  Deploy to production (services + nginx)"
	@echo ""
	@echo "  Other:"
	@echo "    make help            Show this help message"

BUILD_DIR := build
FRONTEND_DIR := frontend
VENV_DIR := venv
MODELS_DIR := contrib/emotiefflib/models
CONFIG_DIR := config
NGINX_SOURCE := $(CONFIG_DIR)/nginx
NGINX_TARGET := /etc/nginx/sites-available/emotion-ai
SERVICE_SOURCE := $(CONFIG_DIR)/service
SERVICE_TARGET := /etc/systemd/system/emotion-ai.service

install:
	bash install_deps.sh

configure:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake .. -G Ninja

build_backend:
	cd $(BUILD_DIR) && ninja -j4

build_frontend:
	cd $(FRONTEND_DIR) && npm install && npm run build

build: configure build_backend build_frontend

up:
	docker compose up -d

down:
	docker compose down

restart: down up

unit_tests:
	cd build && ./tests/EmotionAI_UnitTests

integration_tests:
	cd build && ./tests/EmotionAI_IntegrationTests

python_env: $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install -r requirements.txt

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

models: python_env
	. $(VENV_DIR)/bin/activate && cd $(MODELS_DIR) && python3 prepare_models_for_emotieffcpplib.py

clean-pyc:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean:
	rm -rf \
		$(BUILD_DIR) \
		$(FRONTEND_DIR)/node_modules \
		$(FRONTEND_DIR)/build \
		$(VENV_DIR) \
		package-lock.json