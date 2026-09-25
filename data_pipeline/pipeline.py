"""Module 1 - Data Pipeline: scrape -> clean -> convert -> store -> query.

Run from the repository root or from this folder:
    python data_pipeline/pipeline.py

Outputs (all regenerated on every run):
    data/raw_books.csv       raw scraped text, exactly as listed on the site
    data/clean_books.csv     typed + converted data
    data/books.db            SQLite database (categories + books, PK/FK)
    outputs/query_results.md every SQL query string with its output, plus the
                             pd.read_sql vs pd.merge comparison
"""

import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
# Project-defined fixed baseline rate (not a market rate, no lookup needed).
GBP_TO_INR = 105.50
CATEGORIES = ["Mystery", "Historical Fiction", "Travel", "Poetry"]
RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
DB_PATH = DATA_DIR / "books.db"


# ---------------------------------------------------------------- 1. scrape
def fetch(session, url, retries=3):
    """GET a page and return parsed HTML; retry on network/HTTP errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"  # site is UTF-8; avoids 'Â£' mojibake
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {url}: {exc}")
            time.sleep(1.5 * attempt)


def category_urls(session):
    """Map category name -> first listing page URL, read from the sidebar."""
    soup = fetch(session, BASE_URL)
    links = soup.select("div.side_categories ul li ul li a")
    return {a.get_text(strip=True): urljoin(BASE_URL, a["href"]) for a in links}


def scrape_category(session, name, url):
    """Scrape every book in one category, following 'next' pagination."""
    rows = []
    while url:
        soup = fetch(session, url)
        for card in soup.select("article.product_pod"):
            rating_tag = card.select_one("p.star-rating")
            rating_classes = rating_tag.get("class", []) if rating_tag else []
            rows.append({
                "title": card.h3.a.get("title", "").strip(),
                "price": card.select_one("p.price_color").get_text(strip=True),
                "star_rating": next((c for c in rating_classes if c != "star-rating"), ""),
                "availability": card.select_one("p.availability").get_text(strip=True),
                "category": name,
            })
        next_link = soup.select_one("li.next a")
        url = urljoin(url, next_link["href"]) if next_link else None
        time.sleep(0.3)  # be polite to the practice site
    return rows


def scrape():
    session = requests.Session()
    session.headers["User-Agent"] = "zepto-capstone-student-scraper/1.0"
    urls = category_urls(session)
    rows = []
    for name in CATEGORIES:
        if name not in urls:
            print(f"  category '{name}' not found on site, skipping")
            continue
        found = scrape_category(session, name, urls[name])
        print(f"  {name}: {len(found)} books")
        rows.extend(found)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 2. clean
def parse_price(text):
    match = re.search(r"\d+(?:\.\d+)?", str(text))
    return float(match.group()) if match else None


def parse_rating(text):
    return RATING_WORDS.get(str(text).strip().capitalize())


def parse_in_stock(text):
    text = str(text).strip().lower()
    if "out of stock" in text:
        return False
    if "in stock" in text:
        return True
    return None  # unrecognised wording


def clean(raw):
    """Type the raw text columns.

    Failure policy:
      * price / rating (numeric) -> median imputation. One bad cell should not
        cost us a whole book; the median is robust to the skew in prices.
        An `imputed` flag keeps the fix visible downstream.
      * availability (boolean) -> drop the row. There is no meaningful
        "median" of a yes/no field, and guessing stock status would put
        false information in front of analysts.
      * rows with no title are dropped (nothing to identify them by).
    """
    df = raw.copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["price_gbp"] = df["price"].map(parse_price)
    df["rating"] = df["star_rating"].map(parse_rating)
    df["in_stock"] = df["availability"].map(parse_in_stock)

    before = len(df)
    df = df[(df["title"] != "") & df["in_stock"].notna()].copy()
    dropped = before - len(df)

    df["imputed"] = df["price_gbp"].isna() | df["rating"].isna()
    n_price_na, n_rating_na = df["price_gbp"].isna().sum(), df["rating"].isna().sum()
    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median()).round()

    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    print(f"  dropped {dropped} rows (unparseable availability / no title); "
          f"imputed {n_price_na} prices and {n_rating_na} ratings with the median")
    return df


# --------------------------------------------------------------- 3. convert
def convert(df):
    df = df.copy()
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    return df


# ----------------------------------------------------------------- 4. store
SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

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
"""


def store(df):
    """Create the schema from scratch and insert the data.

    Returns the two in-memory tables so pd.merge can be tested against SQL.
    """
    categories = (pd.DataFrame({"category_name": sorted(df["category"].unique())})
                  .rename_axis("category_id").reset_index())
    categories["category_id"] += 1

    books = df.merge(categories, left_on="category", right_on="category_name")
    books = books[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]]
    books = books.reset_index(drop=True).rename_axis("book_id").reset_index()
    books["book_id"] += 1
    books["in_stock"] = books["in_stock"].astype(int)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO categories VALUES (?, ?)",
                         categories.itertuples(index=False))
        conn.executemany("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?, ?)",
                         books.itertuples(index=False))
    return categories, books


# ----------------------------------------------------------------- 5. query
QUERIES = {
    "Q1 - SELECT / WHERE: five-star books that are in stock": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating = 5 AND in_stock = 1
        ORDER BY title;
    """,
    "Q2 - ORDER BY + LIMIT: the 10 most expensive books": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC, title
        LIMIT 10;
    """,
    "Q3 - DISTINCT: star ratings that actually occur in the catalogue": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,
    "Q4 - BETWEEN + IN: well-rated books (4 or 5 stars) priced 20-30 GBP": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
          AND rating IN (4, 5)
        ORDER BY price_gbp;
    """,
    "Q5 - JOIN: the 3 highest-rated books per category": """
        SELECT category_name, title, rating, price_gbp
        FROM (
            SELECT c.category_name, b.title, b.rating, b.price_gbp,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.category_id
                       ORDER BY b.rating DESC, b.price_gbp DESC, b.book_id
                   ) AS rn
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        WHERE rn <= 3
        ORDER BY category_name, rating DESC, price_gbp DESC;
    """,
    "Q6 - JOIN + GROUP BY: size, average price and stock share per category": """
        SELECT c.category_name,
               COUNT(*)                     AS n_books,
               ROUND(AVG(b.price_gbp), 2)   AS avg_price_gbp,
               ROUND(AVG(b.price_inr), 2)   AS avg_price_inr,
               ROUND(AVG(b.rating), 2)      AS avg_rating,
               SUM(b.in_stock)              AS n_in_stock
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY n_books DESC;
    """,
}
JOIN_QUERY = "Q5 - JOIN: the 3 highest-rated books per category"


def merge_equivalent(categories, books):
    """Reproduce Q5 with pd.merge on the in-memory DataFrames - no SQL."""
    joined = pd.merge(books, categories, on="category_id", how="inner")
    top3 = (joined.sort_values(["category_id", "rating", "price_gbp", "book_id"],
                               ascending=[True, False, False, True])
                  .groupby("category_id").head(3))
    top3 = top3.sort_values(["category_name", "rating", "price_gbp"],
                            ascending=[True, False, False])
    return top3[["category_name", "title", "rating", "price_gbp"]].reset_index(drop=True)


def run_queries(categories, books):
    lines = ["# Module 1 - SQL query results",
             "",
             f"Generated by `pipeline.py` against `data/books.db` "
             f"(conversion rate: 1 GBP = {GBP_TO_INR:.2f} INR).",
             ""]
    results = {}
    with sqlite3.connect(DB_PATH) as conn:
        for name, sql in QUERIES.items():
            result = pd.read_sql(sql, conn)  # every query is read back via pd.read_sql
            results[name] = result
            print(f"\n{name}  ({len(result)} rows)\n{result.to_string(index=False)}")
            lines += [f"## {name}", "", "```sql", sql.strip("\n").rstrip(), "```", "",
                      f"Rows returned: **{len(result)}**", "",
                      result.to_markdown(index=False), ""]

    sql_df = results[JOIN_QUERY]
    merge_df = merge_equivalent(categories, books)
    pd.testing.assert_frame_equal(sql_df, merge_df, check_dtype=False)
    side_by_side = pd.concat({"pd.read_sql (SQL JOIN)": sql_df,
                              "pd.merge (no SQL)": merge_df}, axis=1)
    side_by_side.columns = [f"{src}: {col}" for src, col in side_by_side.columns]
    print(f"\npd.read_sql vs pd.merge for Q5 match: {sql_df.equals(merge_df)}")

    lines += ["## pd.read_sql vs pd.merge - JOIN query (Q5) reproduced without SQL", "",
              "`pd.merge(books, categories, on='category_id')` followed by sorting on "
              "rating DESC, price DESC, book_id and `groupby('category_id').head(3)` "
              "reproduces the SQL window-function JOIN.", "",
              side_by_side.to_markdown(index=False), "",
              "`pd.testing.assert_frame_equal(sql_df, merge_df)` passed - "
              "**both approaches produce identical output.**", ""]
    (OUT_DIR / "query_results.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ main
def main():
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    print("1. Scraping books.toscrape.com ...")
    raw = scrape()
    raw.to_csv(DATA_DIR / "raw_books.csv", index=False)
    print(f"  total scraped: {len(raw)} books across {raw['category'].nunique()} categories")

    print("2. Cleaning ...")
    df = clean(raw)
    print("3. Converting GBP -> INR at the fixed rate ...")
    df = convert(df)
    df.drop(columns=["price", "star_rating", "availability"]).to_csv(
        DATA_DIR / "clean_books.csv", index=False)
    print(df[["price_gbp", "rating", "in_stock", "price_inr"]].dtypes.to_string())

    assert len(df) >= 60 and df["category"].nunique() >= 3, "scope requirement not met"

    print("4. Loading into SQLite ...")
    categories, books = store(df)
    print(f"  {len(categories)} categories, {len(books)} books -> {DB_PATH.name}")

    print("5-6. Running SQL queries and the pd.merge check ...")
    run_queries(categories, books)
    print(f"\nDone. See {OUT_DIR / 'query_results.md'}")


if __name__ == "__main__":
    main()
