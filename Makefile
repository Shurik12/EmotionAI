.PHONY: configure build build_frontend build_backend install run python_env models clean help unit_tests integration_tests deploy_production

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
	@echo "  make deploy_production - Deploy to production (configure services and nginx)"
	@echo "  make clean             - Remove build artifacts, dependencies, and virtual environment"
	@echo "  make help              - Show this help message"

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

build_frontend:
	cd $(FRONTEND_DIR) && npm install && npm run build

configure:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake .. -G Ninja

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

deploy_production: build
	@echo "Deploying to production..."
	@if [ ! -f "$(NGINX_SOURCE)" ]; then \
		echo "Error: Nginx config file not found at $(NGINX_SOURCE)"; \
		exit 1; \
	fi
	@if [ ! -f "$(SERVICE_SOURCE)" ]; then \
		echo "Error: Service file not found at $(SERVICE_SOURCE)"; \
		exit 1; \
	fi
	
	# Copy service file
	sudo cp $(SERVICE_SOURCE) $(SERVICE_TARGET)
	sudo chmod 644 $(SERVICE_TARGET)
	
	# Copy nginx config
	sudo cp $(NGINX_SOURCE) $(NGINX_TARGET)
	sudo chmod 644 $(NGINX_TARGET)
	
	# Enable and link services
	sudo systemctl enable emotion-ai.service
	sudo ln -sf /etc/nginx/sites-available/emotion-ai /etc/nginx/sites-enabled/
	
	# Reload and restart services
	sudo systemctl daemon-reload
	sudo systemctl restart emotion-ai.service
	sudo systemctl reload nginx
	
	@echo "Production deployment completed!"
	@echo "Service status:"
	sudo systemctl status emotion-ai.service --no-pager

production_status:
	@echo "=== EmotionAI Service Status ==="
	sudo systemctl status emotion-ai.service --no-pager
	@echo ""
	@echo "=== Nginx Status ==="
	sudo systemctl status nginx --no-pager
	@echo ""
	@echo "=== Nginx Configuration Test ==="
	sudo nginx -t

clean:
	rm -rf \
		$(BUILD_DIR) \
		$(FRONTEND_DIR)/node_modules \
		$(FRONTEND_DIR)/build \
		$(VENV_DIR) \
		package-lock.json