MODELS_DIR := contrib/emotiefflib/models

.PHONY: help install python_env models configure build_backend build_frontend build benchmark benchmark_focus benchmark_ai_cost test \
        test_integration up up-build down re-up clean clean-data balance \
        build_landing build_compass

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F'## ' \
		'{split($$1,a,":"); printf "  \033[36m%-15s\033[0m %s\n", a[1], $$2}'

install: ## System deps, submodules, libtorch/onnxruntime
	bash install_deps

python_env: ## Create .venv and install Python deps (uses uv, pyproject.toml)
	uv sync

models: python_env ## Export C++ model headers from emotiefflib
	. .venv/bin/activate && cd $(MODELS_DIR) && python3 prepare_models_for_emotieffcpplib.py

configure: ## Configure CMake with Ninja
	cmake -S . -B build -G Ninja

build_backend: configure ## Build the C++ server
	cmake --build build

build_landing: ## Build the landing SPA into frontend/dist (clears the directory)
	cd frontend && npm install && npm run build

build_compass: ## Build the Compass SPA into frontend/dist/compass
	cd compass/frontend && npm ci && npm run build
	test -f frontend/dist/compass/index.html

# Landing must build first: its Vite build empties frontend/dist, which would
# delete compass/. The explicit order also keeps `make -j build` safe.
build_frontend: build_landing ## Build the landing, then Compass on top of it
	$(MAKE) build_compass

build: build_backend build_frontend ## Configure and build everything

benchmark: ## Build the offline benchmarks (tests/benchmark)
	cmake -S . -B build -G Ninja -DBUILD_BENCHMARKS=ON
	cmake --build build --target emotionai_benchmark

benchmark_focus: benchmark ## Run Focus camera benchmarks, save JSON to tests/benchmark/measurements/
	@mkdir -p tests/benchmark/measurements
	@cd tests/benchmark && \
		$(CURDIR)/build/tests/benchmark/emotionai_benchmark \
		--benchmark_filter="BM_Focus" \
		--benchmark_format=console \
		--benchmark_out="measurements/focus-$$(date +%Y%m%d-%H%M%S).json" \
		--benchmark_out_format=json \
		--benchmark_repetitions=1

benchmark_ai_cost: benchmark ## Run LLM token-cost benchmark, save JSON to tests/benchmark/measurements/
	@mkdir -p tests/benchmark/measurements
	@cd tests/benchmark && \
		$(CURDIR)/build/tests/benchmark/emotionai_benchmark \
		--benchmark_filter="BM_AiCost" \
		--benchmark_format=console \
		--benchmark_out="measurements/ai-cost-$$(date +%Y%m%d-%H%M%S).json" \
		--benchmark_out_format=json \
		--benchmark_repetitions=1

test: ## Build and run the unit test suite
	cmake -S . -B build -G Ninja -DBUILD_TESTS=ON -DEMOTIONAI_BUILD_INTEGRATION_TESTS=OFF
	cmake --build build --target emotionai_tests
	./build/tests/emotionai_tests

test_integration: ## Build/run tests incl. integration+e2e (needs DragonflyDB + models)
	cmake -S . -B build -G Ninja -DBUILD_TESTS=ON -DEMOTIONAI_BUILD_INTEGRATION_TESTS=ON
	cmake --build build --target emotionai_tests
	./build/tests/emotionai_tests

up: ## Start containers
	docker compose up -d

up-build: ## Start containers, rebuilding images
	docker compose up -d --build

down: ## Stop containers
	docker compose down

re-up: down up ## Restart containers

balance: ## Check Multitool API balance
	@./balance

clean: ## Remove build output, .venv, node_modules, caches
	rm -rf build frontend/dist frontend/node_modules compass/frontend/node_modules .venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-data: ## Remove runtime server data (uploads, storage, results, logs)
	sudo rm -rf uploads/* storage/* results/* logs/*
	@echo "Server runtime data cleaned."
