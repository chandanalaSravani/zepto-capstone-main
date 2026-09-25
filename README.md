# Zepto Data & AI Platform - AI/ML Capstone

One repository, three connected modules:

| Module | Folder | What it does | Details |
|---|---|---|---|
| 1. Data pipeline (25) | [`data_pipeline/`](data_pipeline/) | scrapes books.toscrape.com -> cleans -> converts GBP->INR -> normalized SQLite -> SQL + pandas queries | [README](data_pipeline/README.md) |
| 2. Analytics pipeline (50) | [`analytics/`](analytics/) | Titanic: profiling, cleaning, EDA data story, then a leakage-safe modeling pipeline, tuning, regression and a saved joblib pipeline | [README](analytics/README.md) |
| 3. Support assistant (25) | [`support_assistant/`](support_assistant/) | RAG over Zepto's 8 policy docs: MiniLM embeddings + ChromaDB, LangGraph intent router, Pydantic-validated JSON, FastAPI + Docker, offline `MOCK_LLM` mode by default | [README](support_assistant/README.md) |

## Setup

This repo uses **one consolidated `requirements.txt`** at the root for all three modules.
It pulls CPU-only PyTorch wheels to keep the install small. The Docker image for module 3
installs a slimmer subset, [`support_assistant/requirements-docker.txt`](support_assistant/requirements-docker.txt).

Tested with Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

A `.devcontainer/` config is included, so the repo also opens directly in GitHub
Codespaces / VS Code Dev Containers.

No paid services, API keys or accounts are needed anywhere.

## Running each module end to end

Run all commands from the repository root.

### 1. Data pipeline

```bash
python data_pipeline/pipeline.py        # scrape (live) -> data/*.csv, data/books.db, outputs/query_results.md
python data_pipeline/test_cleaning.py   # messy-row robustness check
```

### 2. Analytics

```bash
jupyter nbconvert --to notebook --execute --inplace analytics/01_eda.ipynb       # loads Titanic once, saves titanic.csv
jupyter nbconvert --to notebook --execute --inplace analytics/02_modeling.ipynb  # reads titanic.csv, trains + saves pipeline
```

You can also open the notebooks in Jupyter and run them top to bottom, in order. They are
committed with their outputs. `01_eda.ipynb` falls back to the committed `titanic.csv` when it
can't reach the internet.

### 3. Support assistant

```bash
cd support_assistant
python ingest.py                         # embed the 8 docs into ChromaDB (also done automatically on API startup)
uvicorn main:app --port 7860             # MOCK_LLM defaults to mock mode - no LLM, no key, no network
curl -X POST localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "What is the delivery fee?"}'

# or in Docker
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

The first `ingest.py` run downloads the open-source `all-MiniLM-L6-v2` model (~90 MB, no
account needed). After that, everything runs offline.

## Design decisions

### Module 1: data pipeline

- **Scope:** every book in 4 whole categories (Mystery, Historical Fiction, Poetry, Travel),
  following pagination, which gives 88 books. Scraping whole categories rather than the first
  N listing pages means each category is complete, so per-category SQL results are
  meaningful. Requests are polite: one session, a timeout, retries with backoff, and a delay
  between pages.
- **Cleaning policy:** numeric fields that fail to parse (`price`, `rating`) are
  **median-imputed** and flagged in an `imputed` column. Rows with unparseable `availability`
  or no title are **dropped**, because a yes/no stock value can't be meaningfully imputed and
  a guess would mislead analysts. `test_cleaning.py` proves messy rows never crash the
  pipeline.
- **Currency:** `price_inr = price_gbp x 105.50`, the project's fixed rate
  (**1 GBP = 105.50 INR**). No API call is made, and the rate has no date.
- **Schema:** `categories(category_id PK, category_name UNIQUE)` and
  `books(book_id PK, ..., category_id FK)`, with `CHECK` constraints on `rating` (1-5) and
  `in_stock` (0/1). The script rebuilds the database from scratch on every run.
- **Queries:** 6 queries cover `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN` + `IN`, and
  two `JOIN`s (one with a window function). All are read back with `pd.read_sql`. The
  top-3-per-category JOIN is reproduced with `pd.merge` + `groupby().head(3)`, and the two are
  verified identical with `assert_frame_equal`.

### Module 2: analytics

- **One load:** `sns.load_dataset` is called once, in `01_eda.ipynb`, and saved immediately
  as `titanic.csv`. `02_modeling.ipynb` reads only that CSV.
- **Missing values:** handled by the threshold rule. `embarked` (0.22%): drop rows. `age`
  (19.87%): impute the (`pclass`, `sex`) group median. `deck` (77.22%): keep as an
  `"Unknown"` category, because missingness itself predicts survival (29.9% vs 66.7%).
- **No leakage:** the modeling notebook reuses only the structural cleaning. Imputation,
  encoding and scaling live in a `ColumnTransformer` inside a `Pipeline`, fit on the
  stratified training split only. SMOTE runs inside an `imblearn` pipeline, so it only touches
  training folds. The `alive` column (the target restated as text) is excluded.
- **Model choice:** the tuned Random Forest (`n_estimators=400, max_depth=8,
  max_features=None`, OOB 0.833) is deployed. It has the best test accuracy (0.831),
  precision (0.828) and F1 (0.762). Logistic Regression keeps the best AUC (0.861) and is the
  alternative if explainability matters most.
- **Artifact:** the whole fitted pipeline, preprocessing included, is saved with `joblib` and
  verified to work on raw passengers that have missing values.

### Module 3: support assistant

- **Chunking:** one chunk per document. The documents are short (55-90 words) and each covers
  one topic, so splitting further would only separate related facts.
- **Embeddings:** local `all-MiniLM-L6-v2` vectors, normalized, in a cosine-distance ChromaDB
  collection (`zepto_policies`).
- **`MOCK_LLM` (default mock):** only the **generation** step inside each LangGraph node
  branches on it. Routing and retrieval are identical in both modes, so the graded mock path
  exercises the real retrieval pipeline.
- **Response schema:** a single Pydantic `AskResponse` model is the graph's output and the
  FastAPI `response_model`. On the optional real-LLM path, invalid JSON, or cited sources
  that weren't retrieved, trigger up to 2 corrective retries before a marked error response.
- **Docker:** the image downloads the embedding model and builds the index at build time, so
  the running container needs no network.

## Git workflow

Each module was built on its own feature branch (`feature/data-pipeline`,
`feature/analytics`, `feature/support-assistant`). Each branch has several commits and was
merged back into `main` with a merge commit (`git merge --no-ff`), so the history shows up in
`git log --graph --all`.
