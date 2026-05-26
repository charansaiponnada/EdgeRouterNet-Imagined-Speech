# Makefile for BCI Imagined Speech Training

DATA_DIR ?= ./data/mock_dataset
OUT_DIR ?= ./runs
MODE ?= both
TAG ?= BCI_RUN

.PHONY: run setup data-mock clean test lint

run: setup
	python run.py --data_dir $(DATA_DIR) --out_dir $(OUT_DIR) --mode $(MODE) --tag $(TAG)

setup:
	pip install -r requirements.txt

data-mock:
	python src/generate_mock_data.py --out_dir $(DATA_DIR)

test:
	pytest tests/

lint:
	ruff check .

visualize:
	python src/visualize.py --ckpt $(CKPT) --data_file $(DATA_FILE)

clean:
	rm -rf $(OUT_DIR)
	rm -rf $(DATA_DIR)
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
