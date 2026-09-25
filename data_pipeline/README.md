# Module 1 - Data Pipeline (`/data_pipeline`)

Scrape -> clean -> convert -> store -> query, in one script: [`pipeline.py`](pipeline.py).

## Run

```bash
# from the repository root, with the consolidated requirements installed
python data_pipeline/pipeline.py      # scrapes live, rebuilds everything below
python data_pipeline/test_cleaning.py # proves cleaning survives messy rows
```

The script needs network access to `books.toscrape.com` only. Every output is regenerated from
scratch on each run:

| File | Contents |
|---|---|
| `data/raw_books.csv` | raw scraped text, exactly as listed on the site |
| `data/clean_books.csv` | typed and converted data |
| `data/books.db` | SQLite database (`categories` + `books`, PK/FK) |
| `outputs/query_results.md` | every SQL query string with its output, and the `pd.read_sql` vs `pd.merge` comparison |

## What it does

1. **Scrape** (`requests` + `BeautifulSoup`): reads the category links from the site's
   sidebar, then scrapes **every** book in 4 categories, following "next" pagination:
   Mystery (32), Historical Fiction (26), Poetry (19), Travel (11) = **88 books**.
   For each book it captures `title`, `price` (as listed, e.g. `£47.82`), `star_rating`
   (as text, e.g. `Three`), `availability` (as listed text) and `category`. Requests use a
   session, a 15 s timeout, 3 retries with backoff, and a 0.3 s pause between pages.
2. **Clean** into typed columns:
   `price_gbp` (float, currency symbol stripped by regex), `rating` (int 1-5, mapped from
   `One`...`Five`), `in_stock` (bool, parsed from the availability text).
3. **Convert**: `price_inr = round(price_gbp * 105.50, 2)`.
4. **Store**: a normalized two-table SQLite schema (below), created from scratch each run.
5. **Query**: 6 SQL queries, all read back with `pd.read_sql`; the JOIN query is then
   reproduced with `pd.merge` on the in-memory DataFrames and checked with
   `pd.testing.assert_frame_equal`.

## Currency conversion

**1 GBP = 105.50 INR**. This is the project's fixed baseline rate: an artificial constant
defined for this assignment, not a live or historical market rate. No API call is made, and
the rate has no date attached to it.

## Cleaning decisions (what happens to messy rows)

The live site parses cleanly (0 rows dropped, 0 values imputed on the recorded run). The
pipeline still has an explicit policy for bad values, so it never crashes on them:

| Field fails to parse | Action | Why |
|---|---|---|
| `price` (numeric) | **median imputation** | One bad cell shouldn't cost a whole book. The median is robust to the right-skew in prices. |
| `star_rating` (numeric 1-5) | **median imputation**, rounded to an int | Keeps the row and stays within the valid 1-5 range. |
| `availability` (boolean) | **drop the row** | A yes/no field has no meaningful median, and guessing stock status would put false information in front of analysts. |
| empty `title` | **drop the row** | Nothing left to identify the book by. |

An `imputed` flag column records which rows were patched, so the fix stays visible
downstream. [`test_cleaning.py`](test_cleaning.py) feeds `clean()` deliberately broken rows
(`N/A` price, `Seven` stars, `Ask in store` availability, blank title) and asserts the
outcomes above.

## Schema

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);
CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```

The category name is stored once in `categories` and referenced by key from `books`, so
books don't repeat the category text. `PRAGMA foreign_keys = ON` is set before inserting.
The `CHECK` constraints reject out-of-range ratings and non-boolean stock values at the
database level. SQLite has no boolean type, so `in_stock` is stored as 0/1.

## Queries (full text and output in [`outputs/query_results.md`](outputs/query_results.md))

| # | Purpose | Clauses shown | Rows |
|---|---|---|---|
| Q1 | five-star books that are in stock | `SELECT` / `WHERE` | 20 |
| Q2 | 10 most expensive books | `ORDER BY`, `LIMIT` | 10 |
| Q3 | star ratings that occur | `DISTINCT` | 5 |
| Q4 | 4-5 star books priced 20-30 GBP | `BETWEEN`, `IN` | 12 |
| Q5 | 3 highest-rated books per category | `JOIN` (+ window function) | 12 |
| Q6 | size, average price, stock per category | `JOIN`, `GROUP BY` | 4 |

**`pd.read_sql` vs `pd.merge`:** all 6 queries are read into DataFrames with
`pd.read_sql`. Q5 is then rebuilt without SQL:
`pd.merge(books, categories, on="category_id")`, sorted by rating desc, price desc and
book_id (the same tie-breaks as the SQL `ROW_NUMBER()`), then `groupby("category_id").head(3)`.
The two results appear side by side at the end of `query_results.md`, and
`assert_frame_equal` confirms they're identical.
