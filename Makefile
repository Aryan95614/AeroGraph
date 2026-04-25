.PHONY: install repro verify-space deploy-space ingest extract build embed serve eval demo paper all clean test dashboard stats pipeline spaces upload-dataset

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

verify-space:
	bash scripts/verify_space.sh

# deploy-space is gated on verify-space passing. Skipping verification
# (e.g. during CI bootstrap) requires `SKIP_VERIFY=1 make deploy-space`.
deploy-space: $(if $(SKIP_VERIFY),,verify-space)
	python scripts/deploy_hf_space.py

upload-dataset:
	python scripts/upload_dataset.py

all: install ingest build embed eval paper

repro:
	@echo "[repro] installing package..."
	pip install -e ".[dev]" >/dev/null
	@echo "[repro] checking data artifacts..."
	@test -f data/graphs/aerograph.pkl || (echo "missing graph — fetch from HF dataset first"; exit 1)
	@test -d data/chroma_db || (echo "missing chroma_db — fetch from HF dataset first"; exit 1)
	@echo "[repro] running integration test..."
	python -m pytest tests/test_integration.py -v
	@echo "[repro] launching cached-mode Gradio demo on :7860 (unset ANTHROPIC_API_KEY) ..."
	unset ANTHROPIC_API_KEY && python app.py
