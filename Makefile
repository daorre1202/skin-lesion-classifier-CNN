.PHONY: help install test clean train resume visualize

help:
	@echo ""
	@echo "  Skin Lesion Classifier"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install              Install all dependencies"
	@echo "  make test                 Run all 46 unit tests"
	@echo "  make clean                Remove cache and temp files"
	@echo ""
	@echo "  make train                Train from scratch (seed 42)"
	@echo "  make train SEED=7         Train with a different seed"
	@echo "  make resume DIR=<path>    Resume from checkpoint directory"
	@echo ""
	@echo "  make visualize DIR=<path> Visualise ensemble output"
	@echo "                            Example: make visualize DIR=outputs/2026-05-20_13-15-53"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .
	@echo "✓ Dependencies installed"

test:
	pytest tests/ -v --tb=short

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage coverage.xml
	@echo "✓ Cache cleared"

SEED   ?= 42
CONFIG ?= configs/config.yaml

train:
	python scripts/train.py --config $(CONFIG) --seed $(SEED)

resume:
	@if [ -z "$(DIR)" ]; then \
	    echo "Error: specify DIR=<checkpoint_directory>"; exit 1; \
	fi
	python scripts/train.py --config $(CONFIG) --seed $(SEED) --resume $(DIR)

visualize:
	@if [ -z "$(DIR)" ]; then \
	    echo "Error: specify DIR=<checkpoint_directory>"; \
	    echo "Example: make visualize DIR=outputs/2026-05-20_13-15-53"; \
	    exit 1; \
	fi
	python scripts/visualize_ensemble.py --checkpoint $(DIR)
