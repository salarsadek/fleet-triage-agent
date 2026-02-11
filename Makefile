SHELL := /bin/bash
CONFIG := config.yaml
PY := python

.PHONY: help install data validate eda train triage aliases deck report app test lint format clean

help:
    @echo "Targets:"
    @echo "  make install   - create .venv and install deps (Linux/macOS)"
    @echo "  make data      - generate synthetic data"
    @echo "  make validate  - run data quality validation (Great Expectations)"
    @echo "  make eda       - run EDA + stats (tables + KM + Cox)"
    @echo "  make train     - train models (risk + subsystem) and log to MLflow"
    @echo "  make triage    - export triage snapshot artifacts"
    @echo "  make aliases   - create *_latest aliases"
    @echo "  make deck      - generate report/auto_deck.pptx from artifacts"
    @echo "  make report    - full pipeline -> validate -> eda -> train -> triage -> aliases -> deck"
    @echo "  make app       - run Streamlit GUI"
    @echo "  make test      - run tests"
    @echo "  make lint      - ruff lint"
    @echo "  make format    - black format"
    @echo "  make clean     - remove generated outputs (keeps raw data .gitkeep)"

install:
    python -m venv .venv
    . .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

data:
    $(PY) -m src.data.simulate --config $(CONFIG)

validate:
    $(PY) -m src.data.validate --config $(CONFIG)

eda:
    $(PY) -m src.reporting.eda_stats --config $(CONFIG)

train:
    $(PY) -m src.models.train --config $(CONFIG)

triage:
    $(PY) -m src.reporting.triage_snapshot --config $(CONFIG) --horizon 30 --model hgb --k 10 --ranking cost --evidence 5 --evidence_topn 5

aliases:
    $(PY) -m src.reporting.make_latest_aliases --config $(CONFIG) --top_similar 5

deck:
    $(PY) -m src.reporting.make_auto_deck --config $(CONFIG) --out report/auto_deck.pptx

report:
    $(PY) -m src.data.validate --config $(CONFIG)
    $(PY) -m src.reporting.eda_stats --config $(CONFIG)
    $(PY) -m src.models.train --config $(CONFIG)
    $(PY) -m src.reporting.triage_snapshot --config $(CONFIG) --horizon 30 --model hgb --k 10 --ranking cost --evidence 5 --evidence_topn 5
    $(PY) -m src.reporting.make_latest_aliases --config $(CONFIG) --top_similar 5
    $(PY) -m src.reporting.make_auto_deck --config $(CONFIG) --out report/auto_deck.pptx

app:
    $(PY) -m streamlit run app/streamlit_app.py

test:
    $(PY) -m pytest -q

lint:
    $(PY) -m ruff check .

format:
    $(PY) -m black .

clean:
    @echo "Cleaning outputs (keeping folder structure)..."
    rm -rf outputs/figures/* outputs/tables/* outputs/reports/*
    touch outputs/figures/.gitkeep outputs/tables/.gitkeep outputs/reports/.gitkeep