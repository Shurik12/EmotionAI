MODELS_DIR := contrib/emotiefflib/models

.PHONY: help install python_env models configure build_backend build_frontend build test \
        up up-build down restart clean

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F'## ' \
		'{split($$1,a,":"); printf "  \033[36m%-15s\033[0m %s\n", a[1], $$2}'

install: ## System deps, submodules, libtorch/onnxruntime
	bash install_deps.sh

python_env: ## Create venv and install Python deps
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

models: python_env ## Export C++ model headers from emotiefflib
	. venv/bin/activate && cd $(MODELS_DIR) && python3 prepare_models_for_emotieffcpplib.py

configure: ## Configure CMake with Ninja
	cmake -S . -B build -G Ninja

build_backend: configure ## Build the C++ server
	cmake --build build

build_frontend: ## Build the React app into frontend/dist
	cd frontend && npm install && npm run build

build: build_backend build_frontend ## Configure and build everything

test: ## Build and run tests; needs CMakeLists.txt:211-213 uncommented
	cmake -S . -B build -G Ninja -DBUILD_TESTS=ON
	cmake --build build --target emotionai_tests
	./build/tests/emotionai_tests

up: ## Start containers
	docker compose up -d

up-build: ## Start containers, rebuilding images
	docker compose up -d --build

down: ## Stop containers
	docker compose down

restart: down up ## Restart containers

clean: ## Remove build output, venv, node_modules, caches
	rm -rf build frontend/dist frontend/node_modules venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
