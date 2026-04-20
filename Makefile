.PHONY: install ingest extract build embed serve eval demo paper all clean test dashboard stats pipeline spaces deploy upload-dataset

install:
	pip install -e ".[dev]"

ingest:
	python -m aerograph ingest

extract:
	python -m aerograph extract

build:
	python scripts/build_graph.py

embed:
	python -m aerograph embed

pipeline:
	python -m aerograph pipeline

serve:
	python -m aerograph serve

eval:
	python scripts/run_eval.py

demo:
	python -m aerograph demo

paper:
	python -m aerograph eval --figures-only
	python scripts/populate_paper.py

stats:
	python -m aerograph stats

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/processed/*.jsonl data/graphs/*.pkl data/chroma_db/
	rm -rf paper/figures/*.png paper/results/*.json

dashboard:
	streamlit run src/aerograph/dashboard.py

spaces:
	python app.py

deploy:
	modal deploy modal_app.py

upload-dataset:
	python scripts/upload_dataset.py

all: install ingest build embed eval paper
