"""Proves the cleaning step survives messy rows without crashing.

The live site parses cleanly, so this feeds clean() deliberately broken rows.
Run:  python data_pipeline/test_cleaning.py
"""

import pandas as pd

from pipeline import clean, convert

messy = pd.DataFrame([
    {"title": "Good row",       "price": "£10.00", "star_rating": "Three", "availability": "In stock",     "category": "Travel"},
    {"title": "Good row 2",     "price": "£30.00", "star_rating": "Five",  "availability": "In stock",     "category": "Travel"},
    {"title": "Bad price",      "price": "N/A",    "star_rating": "Four",  "availability": "In stock",     "category": "Poetry"},
    {"title": "Bad rating",     "price": "£20.00", "star_rating": "Seven", "availability": "Out of stock", "category": "Poetry"},
    {"title": "Bad stock text", "price": "£15.00", "star_rating": "One",   "availability": "Ask in store", "category": "Poetry"},
    {"title": "",               "price": "£12.00", "star_rating": "Two",   "availability": "In stock",     "category": "Poetry"},
])

out = convert(clean(messy))
print(out[["title", "price_gbp", "rating", "in_stock", "price_inr", "imputed"]].to_string(index=False))

assert len(out) == 4, "rows with bad availability / empty title should be dropped"
assert out.loc[out.title == "Bad price", "price_gbp"].item() == 20.0   # median of 10, 30, 20
assert out.loc[out.title == "Bad rating", "rating"].item() == 4        # median of 3, 5, 4 (rounded)
assert not out.loc[out.title == "Bad rating", "in_stock"].item()     # "Out of stock"
assert out["imputed"].sum() == 2
assert out["price_inr"].tolist() == [round(p * 105.50, 2) for p in out["price_gbp"]]
print("\nAll cleaning checks passed.")
