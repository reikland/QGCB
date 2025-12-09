# QGCB – Metaculus Proto Question Generator

Cette app Streamlit génère et évalue des proto-questions de forecasting (pipeline mutations → sources → questions) en s'appuyant sur OpenRouter.

## Prérequis
- Python 3.10+
- Une clé OpenRouter (`OPENROUTER_API_KEY`)

## Installation rapide
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'app Streamlit
1. Exportez votre clé (ou saisissez-la dans la sidebar après lancement) :
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```
2. Démarrez Streamlit depuis la racine du repo :
   ```bash
   streamlit run app.py
   ```
3. Dans l'UI, choisissez vos modèles (main/judge) ou gardez les valeurs par défaut, puis cliquez sur **Run full pipeline**.

> Inutile de créer une nouvelle app ou de pointer sur `__init__` : sélectionnez simplement `app.py` comme entrypoint.

## Tests rapides
```bash
python -m py_compile app.py qgcb/*.py
```

## Structure
- `app.py` : interface Streamlit.
- `qgcb/` : logique métier (prompts, pipeline, OpenRouter helpers, modèles Pydantic, config).
