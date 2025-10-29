.PHONY: configure build build_frontend build_backend install run python_env models clean help unit_tests integration_tests

# Default target
help:
	@echo "Available commands:"
	@echo "  make install           - Install system dependencies"
	@echo "  make python_env        - Create Python virtual environment and install dependencies"
	@echo "  make models            - Generate C++ models from Python scripts"
	@echo "  make configure         - Configure CMake build system"
	@echo "  make build_backend     - Build backend C++ code"
	@echo "  make build_frontend    - Build frontend React application"
	@echo "  make build             - Full build (configure, backend, and frontend)"
	@echo "  make run               - Run the EmotionAI application"
	@echo "  make unit_tests        - Run unit tests"
	@echo "  make integration_tests - Run integration tests"
	@echo "  make clean             - Remove build artifacts, dependencies, and virtual environment"
	@echo "  make help              - Show this help message"

# Variables
BUILD_DIR := build
FRONTEND_DIR := frontend
VENV_DIR := venv
MODELS_DIR := contrib/emotiefflib/models

# Dependencies
install:
	bash install_deps.sh

build_frontend:
	cd $(FRONTEND_DIR) && npm install && npm run build

configure:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake .. -G Ninja -DBUILD_TESTS=ON

build_backend:
	cd $(BUILD_DIR) && ninja -j4

run:
	cd $(BUILD_DIR) && ./EmotionAI

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

build: configure build_backend build_frontend

# Cleanup
clean:
	rm -rf \
		$(BUILD_DIR) \
		$(FRONTEND_DIR)/node_modules \
		$(FRONTEND_DIR)/build \
		$(VENV_DIR) \
		package-lock.json