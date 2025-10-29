.PHONY: configure build build_frontend build_backend install run python_env models clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make python_env    - Setup python environment"
	@echo "  make build         - Build project"
	@echo "  make models        - Create c++ models"
	@echo "  make clean         - Remove build frontend and backend"
	@echo "  make help          - Show this help message"

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