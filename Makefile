.PHONY: install ingest build serve eval demo paper all clean test

install:
	pip install -e ".[dev]"

ingest:
	python -m aerograph.ingest

build:
	python scripts/build_graph.py

serve:
	uvicorn aerograph.api:app --host 0.0.0.0 --port 8000 --reload

eval:
	python scripts/run_eval.py

demo:
	python scripts/demo.py

paper:
	python -m aerograph.eval --figures-only

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/processed/*.jsonl data/graphs/*.pkl data/chroma_db/
	rm -rf paper/figures/*.png paper/results/*.json

all: install ingest build eval paper
