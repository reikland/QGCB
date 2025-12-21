# QGCB – Metaculus Proto Question Generator

This Streamlit app generates and evaluates proto-questions for forecasting (pipeline: mutations → sources → questions) using OpenRouter.

## Prerequisites
- Python 3.10+
- An OpenRouter key (`OPENROUTER_API_KEY`)

## Quick Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App
1. Export your key (or enter it in the sidebar after launch):
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```
2. Start Streamlit from the repo root:
   ```bash
   streamlit run app.py
   ```
3. In the UI, choose your models (main/judge) or keep the defaults, then click **Run full pipeline**.

> No need to create a new app or point to `__init__`: simply select `app.py` as the entrypoint.

## Quick Tests
```bash
python -m py_compile app.py qgcb/*.py
```

## Structure
- `app.py`: Streamlit interface.
- `qgcb/`: business logic (prompts, pipeline, OpenRouter helpers, Pydantic models, config).
