"""Generate 01_eda.ipynb and 02_modeling.ipynb from the cell lists below.

Keeping the notebook source as plain Python makes it easy to diff and review.
Regenerate + execute from the repository root:
    python analytics/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace analytics/01_eda.ipynb
    jupyter nbconvert --to notebook --execute --inplace analytics/02_modeling.ipynb
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


# =====================================================================
# 01_eda.ipynb
# =====================================================================
EDA = [
md("""
# 01 - Titanic EDA: profile, clean, and tell the data story

Module 2, Part A. This notebook is the **only** place the raw dataset is loaded
(`sns.load_dataset('titanic')`). It is saved straight away as `titanic.csv`, the committed offline
fallback, and `02_modeling.ipynb` continues from that same file.
"""),
code("""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(FIG / name, dpi=110, bbox_inches="tight")
"""),
md("""
## Task 1 - Load once, save the offline fallback, profile

`sns.load_dataset` needs internet the first time it runs. If that fails (for example, grading
offline), the committed `titanic.csv` is read instead. Either way this is the module's single
load of the raw data.
"""),
code("""
try:
    df = sns.load_dataset("titanic")          # the one and only raw load
    source = "sns.load_dataset('titanic')"
except Exception as exc:                      # offline: fall back to the committed copy
    df = pd.read_csv("titanic.csv")
    source = f"titanic.csv (offline fallback: {type(exc).__name__})"

df.to_csv("titanic.csv", index=False)         # committed offline fallback
print("Loaded from:", source)
print("Shape:", df.shape)
"""),
code("df.info()"),
code("df.describe()"),
code("df.head()"),
code("""
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = (pd.DataFrame({"n_missing": missing, "pct_missing": missing_pct})
                  .query("n_missing > 0")
                  .sort_values("pct_missing", ascending=False))
missing_report
"""),
code("""
balance = df["survived"].value_counts().rename(index={0: "died (0)", 1: "survived (1)"})
print(balance.to_string())
print((df["survived"].value_counts(normalize=True) * 100).round(2).rename("pct").to_string())
"""),
md("""
**Profile summary.** The dataset has 891 rows and 15 columns. Four columns have missing values:
`deck` **77.22%**, `age` **19.87%**, `embarked` **0.22%** and `embark_town` **0.22%**
(`embarked` and `embark_town` are missing on the same 2 rows).
The target is imbalanced: **549 died (61.62%)** and **342 survived (38.38%)**, roughly 1.6 : 1.
That imbalance is why the modeling notebook uses a stratified split and compares imbalance
strategies.
"""),
md("""
## Task 2 - Missing-value handling (threshold rule)

Rule: **under 5% missing -> drop those rows; 5-30% -> impute; above 30% -> imputation is
unreliable, so either drop the column or encode "missing" as its own category.**
"""),
code("""
def strategy(pct):
    if pct < 5:
        return "drop rows (< 5%)"
    if pct <= 30:
        return "impute (5-30%)"
    return "too high to impute (> 30%): drop column or 'missing' category"

missing_report.assign(strategy=missing_report["pct_missing"].map(strategy))
"""),
md("""
**Decision for each affected column, citing the measured percentage and the rule:**

| Column | Measured missing | Rule band | Decision |
|---|---|---|---|
| `embarked` | 0.22% (2 rows) | < 5% -> drop rows | Drop the 2 rows. Losing 0.22% of the data costs nothing, and imputing a port would be a guess. |
| `embark_town` | 0.22% (the same 2 rows) | < 5% -> drop rows | Removed by the same row drop. |
| `age` | 19.87% (177 rows) | 5-30% -> impute | Impute the **median age within each (`pclass`, `sex`) group**. The median suits a right-skewed variable better than the mean. Grouping matters because age differs sharply by class: the median is 40 for 1st-class men and 21.5 for 3rd-class women. One global median (28) would pull every imputed passenger towards the middle. An `age_was_missing` flag keeps the imputation traceable. |
| `deck` | 77.22% (688 rows) | > 30% -> too high to impute | **Encode missing as its own category, `"Unknown"`**, rather than dropping the column. With over three quarters of values absent, any imputed deck would mostly be invented. But whether a deck is missing is itself informative: passengers with a recorded deck survived at **66.7%**, versus **29.9%** for "Unknown", because deck records exist mostly for 1st-class cabins. Dropping the column would throw that signal away, while guessing a deck would create false data. |

After cleaning: **889 rows, 0 missing values.**
"""),
code("""
clean = df.copy()

# embarked / embark_town: 0.22% missing each (the same 2 rows) -> drop rows
clean = clean.dropna(subset=["embarked", "embark_town"])

# age: 19.87% missing -> impute with the median age of the passenger's (pclass, sex) group
group_median = clean.groupby(["pclass", "sex"])["age"].transform("median")
clean["age_was_missing"] = clean["age"].isna()
clean["age"] = clean["age"].fillna(group_median)

# deck: 77.22% missing -> keep the column, encode missing as its own category "Unknown"
clean["deck"] = clean["deck"].astype("object").fillna("Unknown")

print("rows before:", len(df), "| rows after:", len(clean))
print("remaining missing values:", int(clean.isna().sum().sum()))
clean.groupby(["pclass", "sex"])["age"].median().rename("median age used for imputation")
"""),
code("""
# does 'deck unknown' carry information? (why we kept it as a category)
clean.assign(deck_known=clean["deck"] != "Unknown").groupby("deck_known")[["survived"]].mean().round(3)
"""),
md("""
## Task 3 - Univariate analysis: `age` and `fare`
"""),
code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for row, col in enumerate(["age", "fare"]):
    sns.histplot(clean[col], bins=40, kde=True, ax=axes[row, 0], color="#4C72B0")
    axes[row, 0].set_title(f"{col} - histogram")
    sns.boxplot(x=clean[col], ax=axes[row, 1], color="#DD8452")
    axes[row, 1].set_title(f"{col} - box plot")
fig.tight_layout()
save(fig, "01_univariate_age_fare.png")
plt.show()
"""),
code("""
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {"Q1": q1, "Q3": q3, "IQR": iqr, "lower_fence": lo, "upper_fence": hi,
            "n_below": int((s < lo).sum()), "n_above": int((s > hi).sum()),
            "n_outliers": int(((s < lo) | (s > hi)).sum())}

outliers = pd.DataFrame({
    "age (cleaned, imputed)": iqr_outliers(clean["age"]),
    "age (raw, non-missing only)": iqr_outliers(df["age"].dropna()),
    "fare (cleaned)": iqr_outliers(clean["fare"]),
}).T.round(2)
outliers
"""),
code("""
fare_stats = pd.Series({
    "mean": clean["fare"].mean(),
    "median": clean["fare"].median(),
    "mode": clean["fare"].mode().iloc[0],
    "skewness": clean["fare"].skew(),
}).round(3)
fare_stats
"""),
md("""
**IQR outliers.**
- **`fare`:** **114 outliers**, all above the upper fence of 65.66. None fall below the lower
  fence, which is negative.
- **`age`:** **32 outliers** on the cleaned (imputed) column, all above 57.75. On the raw,
  non-missing ages there are only **11** (above 64.81). Imputing 177 values at group medians
  piles values up in the middle, which narrows the IQR, so more old passengers fall outside
  the fences. Both counts are shown so the effect of imputation is visible.

**Skewness of `fare`.** **mode (8.05) < median (14.45) < mean (32.10)**. That ordering is the
signature of a **right-skewed** distribution, and the skewness coefficient of +4.80 confirms it.
Most passengers paid low 3rd-class fares, while a long tail of expensive 1st-class tickets
(up to 512.33) pulls the mean far above the median.
"""),
md("""
## Task 4 - Bivariate analysis

Survival rates are computed with explicit boolean masks, combining conditions with `&`.
"""),
code("""
rate = lambda mask: clean.loc[mask, "survived"].mean()

by_sex = pd.Series({s: rate(clean["sex"] == s) for s in ["female", "male"]}, name="survival_rate")
by_class = pd.Series({c: rate(clean["pclass"] == c) for c in [1, 2, 3]}, name="survival_rate")
by_sex_class = pd.DataFrame(
    [{"sex": s, "pclass": c,
      "n": int(((clean["sex"] == s) & (clean["pclass"] == c)).sum()),
      "survival_rate": rate((clean["sex"] == s) & (clean["pclass"] == c))}
     for s in ["female", "male"] for c in [1, 2, 3]])

print("(a) by sex\\n", by_sex.round(3).to_string(), "\\n")
print("(b) by pclass\\n", by_class.round(3).to_string(), "\\n")
print("(c) by sex AND pclass\\n", by_sex_class.round(3).to_string(index=False))
"""),
code("""
# an | example: women OR children (< 16) vs everyone else
women_or_children = (clean["sex"] == "female") | (clean["age"] < 16)
print(f"women or children: {rate(women_or_children):.3f}   adult men: {rate(~women_or_children):.3f}")
"""),
code("""
CORR_COLS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]   # adult_male / alone excluded
corr = clean[CORR_COLS].corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Correlation matrix (6 numeric columns)")
save(fig, "02_correlation_heatmap.png")
plt.show()

pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))   # upper triangle, no diagonal
             .stack().dropna().rename("r").to_frame())
pairs["abs_r"] = pairs["r"].abs()
pairs.sort_values("abs_r", ascending=False).round(3)
"""),
md("""
**Survival rates (boolean masks):**
- **(a) by sex:** female **74.0%**, male **18.9%**.
- **(b) by class:** 1st **62.6%**, 2nd **47.3%**, 3rd **24.2%**.
- **(c) by sex and class:** female 1st **96.7%**, female 2nd **92.1%**, female 3rd **50.0%**,
  male 1st **36.9%**, male 2nd **15.7%**, male 3rd **13.5%**.
- Using `|`, "women or children under 16" survived at **71.6%**, versus **16.4%** for adult men.

**The two strongest correlations** (ranked by absolute off-diagonal r, from the sorted table above):
1. **`pclass` vs `fare`, r = -0.548.** Higher class *numbers* (cheaper classes) go with lower
   fares. The ticket price largely *is* the class, so these two columns carry overlapping
   information about wealth and status.
2. **`sibsp` vs `parch`, r = +0.415.** Passengers travelling with siblings or spouses also
   tended to travel with parents or children, meaning whole families travelled together.
   Both columns measure family size from different angles.

The next pair, `pclass` vs `age` (r = -0.411), is a very close third: older passengers were
more often in 1st class. Part of this correlation comes from our group-median imputation,
since the raw non-missing ages give r = -0.366. Among correlations with the target, `pclass`
is the strongest (r = -0.336).
"""),
md("""
## Task 5 - Multivariate data story: who survived, and why?
"""),
code("""
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=clean, x="pclass", y="survived", hue="sex", errorbar=None, ax=ax,
            palette={"female": "#C44E52", "male": "#4C72B0"})
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f")
ax.set(title="Chart 1 - Survival rate by class and sex", ylabel="survival rate", xlabel="passenger class")
save(fig, "03_story_class_sex.png")
plt.show()
"""),
md("""
**Interpretation.** Sex is the dominant factor. In every class, women survived at a far higher
rate than men (96.7% vs 36.9% in 1st class, 50.0% vs 13.5% in 3rd). Class is the second
factor: for both sexes, survival falls steadily from 1st to 3rd class. A 1st-class woman was
about seven times as likely to survive as a 3rd-class man, consistent with a
"women and children first" evacuation in which access to the boat deck favoured the upper
classes.
"""),
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
known_age = df.dropna(subset=["age"])        # raw ages only, so imputed values don't create a spike
for ax, sex in zip(axes, ["female", "male"]):
    sns.histplot(data=known_age[known_age["sex"] == sex], x="age", hue="survived", bins=range(0, 85, 5),
                 multiple="fill", ax=ax, palette={0: "#8C8C8C", 1: "#55A868"})
    ax.set(title=f"{sex}: share surviving by age band", ylabel="share of passengers")
fig.suptitle("Chart 2 - Survival share by age (5-year bands), split by sex")
fig.tight_layout()
save(fig, "04_story_age_sex.png")
plt.show()

child = known_age["age"] < 16
print("male children (<16) survival:", round(known_age.loc[child & (known_age.sex == "male"), "survived"].mean(), 3))
print("adult men (>=16) survival:   ", round(known_age.loc[~child & (known_age.sex == "male"), "survived"].mean(), 3))
"""),
md("""
**Interpretation.** The "children first" part of the rule shows up in the male panel. Boys
under 16 survived at **52.5%**, against **17.4%** for adult men, so being a child mattered
enormously for males. Among females, survival is high at almost every age, so age adds little
on top of sex. This chart uses only raw (non-imputed) ages, so the imputed median values
don't create an artificial spike.
"""),
code("""
fig, ax = plt.subplots(figsize=(9, 5))
plot_df = clean[clean["fare"] > 0]
sns.scatterplot(data=plot_df, x="age", y="fare", hue="survived", style="pclass", alpha=0.7, ax=ax,
                palette={0: "#8C8C8C", 1: "#55A868"})
ax.set_yscale("log")
ax.set(title="Chart 3 - Fare (log scale) vs age, coloured by survival", ylabel="fare (log scale)")
save(fig, "05_story_fare_age.png")
plt.show()

fare_q = pd.qcut(clean["fare"], 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 dearest"])
clean.groupby(fare_q, observed=True)["survived"].mean().round(3).rename("survival rate by fare quartile")
"""),
md("""
**Interpretation.** Survivors (green) cluster in the upper part of the plot, at higher fares,
and 1st-class markers dominate that region. Fare quartiles make the gradient explicit:
survival rises from **19.7%** in the cheapest quartile to **57.7%** in the most expensive.
Fare works as a proxy for class and cabin location. Wealthier passengers were berthed closer
to the lifeboats and could reach them first. (Chart 3 leaves out the 15 zero-fare tickets
because the axis is logarithmic.)
"""),
code("""
clean["family_size"] = clean["sibsp"] + clean["parch"] + 1
fam = clean.groupby("family_size").agg(n=("survived", "size"), survival_rate=("survived", "mean"))

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=fam.index, y=fam["survival_rate"], color="#8172B2", ax=ax)
for i, (n, r) in enumerate(zip(fam["n"], fam["survival_rate"])):
    ax.text(i, r + 0.02, f"n={n}", ha="center", fontsize=9)
ax.set(title="Chart 4 - Survival rate by family size aboard", xlabel="family size (self + sibsp + parch)",
       ylabel="survival rate", ylim=(0, 1))
save(fig, "06_story_family_size.png")
plt.show()
fam.round(3)
"""),
md("""
**Interpretation.** Family size has a non-linear effect. People travelling alone survived at
only **30.1%**. Small families of 2-4 did best, at **55-72%**: a companion could help, and
these groups often included women and children. Large families of 5 or more fared badly
(0-33%). They were mostly 3rd-class households that were hard to keep together and evacuate.
This is why `sibsp` and `parch` matter even though their straight-line correlation with
`survived` is weak (r = -0.03 and 0.08).
"""),
code("""
pivot = clean.pivot_table(index="embark_town", columns="pclass", values="survived", aggfunc="mean")
counts = clean.pivot_table(index="embark_town", columns="pclass", values="survived", aggfunc="size")

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1, ax=ax)
ax.set_title("Chart 5 - Survival rate by embarkation port and class")
save(fig, "07_story_port_class.png")
plt.show()
pd.concat({"survival_rate": pivot.round(3), "n_passengers": counts}, axis=1)
"""),
md("""
**Interpretation.** Cherbourg passengers survived most often overall (55.4%, against 33.7%
for Southampton), but the heatmap shows that is largely a *class-mix* effect. 85 of
Cherbourg's 168 passengers were in 1st class, while Southampton's were mostly 3rd class.
Within the same class the gaps shrink (1st class: 69.4% Cherbourg vs 58.3% Southampton), and
Queenstown's cells rest on very few 1st/2nd-class passengers (2 and 3). The port of
embarkation is therefore mostly a stand-in for class, not an independent cause.
"""),
md("""
### The story in one paragraph

Survival on the Titanic was decided mostly by **who you were and where you were berthed**.
**Sex** is the strongest single factor (74% of women survived vs 19% of men), reflecting the
"women and children first" rule, and **childhood** protected boys too (52.5% vs 17.4% for
adult men). **Class**, and **fare** as its price tag, came second: survival fell from 63% (1st) to 47% (2nd) to
24% (3rd), and it tripled from the cheapest to the dearest fare quartile.
**Family size** helped in small groups and hurt in large ones. The **port of embarkation**
mostly reflects each port's class mix. These findings point the model towards `sex`,
`pclass`, `age`, `fare` and family variables as the key features.
"""),
md("""
## Task 6 - Exploratory z-score standardization check

`z = (x - mean) / std`, computed by hand on the full cleaned DataFrame. This is an EDA sanity
check only. The modeling notebook fits its own `StandardScaler` on the training split.
"""),
code("""
z = pd.DataFrame({f"{c}_z": (clean[c] - clean[c].mean()) / clean[c].std() for c in ["age", "fare"]})

summary = pd.DataFrame({
    "mean_before": clean[["age", "fare"]].mean().values,
    "std_before": clean[["age", "fare"]].std().values,
    "mean_after": z.mean().values,
    "std_after": z.std().values,
}, index=["age", "fare"]).round(4)
summary
"""),
code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for row, c in enumerate(["age", "fare"]):
    sns.histplot(clean[c], bins=40, ax=axes[row, 0], color="#4C72B0")
    axes[row, 0].set_title(f"{c} - before (mean {clean[c].mean():.2f}, std {clean[c].std():.2f})")
    sns.histplot(z[f"{c}_z"], bins=40, ax=axes[row, 1], color="#55A868")
    axes[row, 1].set_title(f"{c} - after z-score (mean {z[f'{c}_z'].mean():.2f}, std {z[f'{c}_z'].std():.2f})")
fig.tight_layout()
save(fig, "08_zscore_before_after.png")
plt.show()
"""),
md("""
**Before/after check.** Before scaling, `age` has mean 29.07 and std 13.27, and `fare` has
mean 32.10 and std 49.70. After `z = (x - mean) / std`, both columns have **mean 0.0000 and
std 1.0000**, which confirms the transformation. The histograms show the *shape* is unchanged:
`fare` is still strongly right-skewed. Standardization only shifts and rescales a variable; it
does not make it normal. This check is exploratory only. The modeling pipeline fits its own
`StandardScaler` on the training split.
"""),
]


# =====================================================================
# 02_modeling.ipynb
# =====================================================================
MODELING = [
md("""
# 02 - Titanic modeling pipeline

Module 2, Part B. This notebook continues from `01_eda.ipynb`. It reads the **same committed
`titanic.csv`** that notebook saved and never calls `sns.load_dataset` again.

The Task 2 structural cleaning decisions are re-applied here because they don't learn anything
from the data:
drop the 2 rows missing `embarked`, and leave `deck` out.

The Task 2 **age imputation is deliberately not re-used**. It computed group medians over the
full dataset, which would leak test-set information into training. Here, age imputation lives
inside the pipeline and is fit on the training split only.
"""),
code("""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

sns.set_theme(style="whitegrid", context="notebook")
SEED = 42
FIG = Path("figures")
MODELS = Path("models")
FIG.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(FIG / name, dpi=110, bbox_inches="tight")
"""),
code("""
df = pd.read_csv("titanic.csv")                      # same file 01_eda.ipynb produced
df = df.dropna(subset=["embarked"])                  # Task 2 rule: < 5% missing -> drop rows

NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "survived"

# Deliberately excluded: 'alive' (the target as text - pure leakage), 'class' / 'embark_town' / 'who'
# (duplicates of pclass / embarked / sex+age), 'adult_male' / 'alone' (derived flags), 'deck' (77% missing).
X, y = df[FEATURES], df[TARGET]
print(X.shape)
print(y.value_counts().to_string())
print((y.value_counts(normalize=True) * 100).round(2).to_string())
"""),
md("""
## Task 7 - Stratified train/test split (before any preprocessing)
"""),
code("""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

split_balance = pd.DataFrame({
    "full": y.value_counts(normalize=True),
    "train": y_train.value_counts(normalize=True),
    "test": y_test.value_counts(normalize=True),
}).round(4)
print(len(X_train), "train rows /", len(X_test), "test rows")
split_balance
"""),
md("""
**Why stratify.** The target is imbalanced: 61.75% died vs 38.25% survived (549 vs 340 after
dropping the 2 rows with no `embarked`). An unstratified 80/20 split of 889 rows could, by
chance, give a test set with noticeably more or fewer survivors, which would skew every test
metric, especially precision and recall for the minority *survived* class. With
`stratify=y`, both splits keep the original ratio: **train 38.26% survived, test 38.20%**. The
178-row test set is then a fair miniature of the population. The split happens **before** any
imputation, encoding or scaling.
"""),
md("""
## Task 8 - Preprocessing, fit on the training split only

One `ColumnTransformer` does per-column imputation, encoding and scaling. It sits inside a
`Pipeline` with each estimator, so `pipeline.fit(X_train, y_train)` fits the imputers, encoder
and scaler on training rows only. `predict(X_test)` then only calls `transform` on the test
rows. The pipeline structure enforces the separation; nothing depends on remembering it by hand.

| Columns | Steps | Why |
|---|---|---|
| `pclass`, `age`, `sibsp`, `parch`, `fare` | `SimpleImputer(median)` -> `StandardScaler` | Median is robust to the skew in `fare`/`age`; only `age` actually has gaps (~20%) but the imputer protects against missing values in any future raw input. `pclass` is kept as an ordinal number (1 < 2 < 3). |
| `sex`, `embarked` | `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")` | One-hot avoids implying an order between ports; `handle_unknown="ignore"` keeps the saved pipeline from crashing on an unseen category. |
"""),
code("""
def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])

def make_pipeline(estimator):
    return Pipeline([("prep", make_preprocessor()), ("clf", estimator)])

# demonstration: the fitted scaler's statistics come from training rows only
demo = make_preprocessor().fit(X_train)
scaler = demo.named_transformers_["num"].named_steps["scale"]
imputer = demo.named_transformers_["num"].named_steps["impute"]
print("imputer medians (train):", dict(zip(NUMERIC, imputer.statistics_.round(2))))
print("scaler means    (train):", dict(zip(NUMERIC, scaler.mean_.round(2))))
print("age median of test rows, NOT used:", X_test["age"].median())
print("output features:", list(demo.get_feature_names_out()))
"""),
md("""
## Task 9 - Three classifiers on the identical split
"""),
code("""
models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
    "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED)),
    "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    print(f"fitted {name}")
"""),
md("""
The Decision Tree is capped at `max_depth=4` (with `min_samples_leaf=5`). An unconstrained tree
memorizes the training set, and a depth-4 tree is also small enough for `plot_tree` to stay
readable.
"""),
code("""
tree_pipe = models["Decision Tree"]
feature_names = [n.split("__", 1)[1] for n in tree_pipe.named_steps["prep"].get_feature_names_out()]

fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(tree_pipe.named_steps["clf"], feature_names=feature_names, class_names=["Died", "Survived"],
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree (max_depth=4) - numeric features are standardized values")
save(fig, "09_decision_tree.png")
plt.show()
"""),
md("""
**Reading the tree.** Numeric thresholds are on standardized values; for example,
`pclass <= 0.21` means `pclass <= 2.49`, i.e. 1st or 2nd class. The root split is `sex_female`,
which carries 63% of the tree's feature importance. Women in 1st/2nd class go to leaves that
predict *Survived*. For men, the next split is `age <= -1.99` (about 3.5 years old), then class.
The tree rediscovers the EDA story on its own: sex first, then class and childhood, with
`pclass` at 19.5% importance and `age` and `fare` at about 8% each.
"""),
md("""
## Task 10 - Evaluation: confusion matrix, accuracy, precision, recall, F1, ROC/AUC
"""),
code("""
def evaluate(pipe, X_eval=X_test, y_eval=y_test):
    pred = pipe.predict(X_eval)
    proba = pipe.predict_proba(X_eval)[:, 1]
    return {"accuracy": accuracy_score(y_eval, pred), "precision": precision_score(y_eval, pred),
            "recall": recall_score(y_eval, pred), "f1": f1_score(y_eval, pred),
            "roc_auc": roc_auc_score(y_eval, proba)}

results = pd.DataFrame({name: evaluate(pipe) for name, pipe in models.items()}).T.round(4)
results
"""),
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, pipe) in zip(axes, models.items()):
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, display_labels=["Died", "Survived"],
                                          cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(name)
fig.suptitle("Confusion matrices (test set, n=%d)" % len(y_test))
fig.tight_layout()
save(fig, "10_confusion_matrices.png")
plt.show()

for name, pipe in models.items():
    tn, fp, fn, tp = confusion_matrix(y_test, pipe.predict(X_test)).ravel()
    print(f"{name:20s} TN={tn:3d} FP={fp:3d} FN={fn:3d} TP={tp:3d}")
"""),
code("""
fig, ax = plt.subplots(figsize=(7, 6))
for name, pipe in models.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="chance (AUC = 0.5)")
ax.set_title("ROC curves (test set)")
ax.legend(loc="lower right")
save(fig, "11_roc_curves.png")
plt.show()
"""),
code("""
# side-by-side comparison table: confusion-matrix counts + every metric
cm_cols = {}
for name, pipe in models.items():
    tn, fp, fn, tp = confusion_matrix(y_test, pipe.predict(X_test)).ravel()
    cm_cols[name] = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
comparison = pd.concat([pd.DataFrame(cm_cols).T, results], axis=1)
comparison
"""),
md("""
**Reading the results.**
- **Logistic Regression:** best **ROC AUC (0.861)** and best precision among the three
  untuned models (0.783).
- **Random Forest (default):** catches slightly more survivors (recall 0.706, 48 TP) at the
  cost of more false alarms (15 FP), and has the lowest AUC (0.824). With no depth limit its
  trees overfit.
- **Decision Tree (depth 4):** close behind at accuracy 0.798 and AUC 0.851. It's a strong
  result for such a small, fully interpretable model.

All three sit at roughly 80% accuracy. Every model misses 20-23 of the 68 survivors, so recall
is the weak spot, consistent with survivors being the minority class.
"""),
md("""
## Task 11 - Class-imbalance handling (Logistic Regression, three ways)
"""),
code("""
print("training-set class balance:")
print(y_train.value_counts().rename(index={0: "died", 1: "survived"}).to_string())
print(f"ratio died:survived = {(y_train == 0).sum() / (y_train == 1).sum():.2f} : 1")
"""),
code("""
variants = {
    "(a) baseline": make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
    "(b) class_weight='balanced'": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    # imblearn's Pipeline applies SMOTE only during fit(), i.e. only to training rows;
    # predict() on the test set skips the sampler entirely.
    "(c) SMOTE (train fold only)": ImbPipeline([
        ("prep", make_preprocessor()),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
}
for pipe in variants.values():
    pipe.fit(X_train, y_train)

smote_pipe = variants["(c) SMOTE (train fold only)"]
Xt = smote_pipe.named_steps["prep"].transform(X_train)
_, y_res = smote_pipe.named_steps["smote"].fit_resample(Xt, y_train)
print("training labels after SMOTE:", np.bincount(y_res), "| test labels untouched:", np.bincount(y_test))

imbalance = pd.DataFrame({n: evaluate(p) for n, p in variants.items()}).T[["precision", "recall", "f1"]]

# a single 179-row test set is noisy, so also report 5-fold stratified CV on the training split
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for n, p in variants.items():
    f1s, recs = [], []
    for tr, va in cv.split(X_train, y_train):
        p.fit(X_train.iloc[tr], y_train.iloc[tr])
        pr = p.predict(X_train.iloc[va])
        f1s.append(f1_score(y_train.iloc[va], pr)); recs.append(recall_score(y_train.iloc[va], pr))
    imbalance.loc[n, "cv_recall"] = np.mean(recs)
    imbalance.loc[n, "cv_f1"] = np.mean(f1s)
    p.fit(X_train, y_train)                            # refit on the full training split
imbalance.round(4)
"""),
md("""
**Class balance (training split):** 439 died vs 272 survived (1.61 : 1). SMOTE, applied only
inside `fit()` on the training rows, balances them to 439 : 439. The 178 test rows are never
resampled.

**Conclusion.** Both imbalance strategies traded precision for recall, exactly as expected.
- **Recall on the minority *survived* class:** rose from 0.691 (baseline) to **0.750** with
  `class_weight='balanced'` and 0.735 with SMOTE.
- **Precision:** fell from 0.783 to 0.718 and 0.735.
- **Test-set F1:** nearly identical across all three (0.734 / 0.734 / 0.735), so a single
  178-row test set can't separate them.
- **5-fold CV on the training split:** here both strategies clearly beat the baseline (CV F1
  0.710 -> 0.727 / 0.728, CV recall 0.691 -> 0.761 / 0.754).

**`class_weight='balanced'` worked best overall.** It gives the highest recall on both test
and CV, and effectively ties SMOTE on F1. It's also simpler: it reweights the loss instead of
creating synthetic passengers, so it adds no interpolated rows that may not look like real
people.

The imbalance here is mild (1.6 : 1), which is why the gains are modest. Which variant to
prefer depends on the cost of errors: if missing a survivor is costlier than a false alarm,
use the balanced weights.
"""),
md("""
## Task 12 - Hyperparameter tuning: GridSearchCV over the Random Forest (with OOB score)
"""),
code("""
rf_pipe = make_pipeline(RandomForestClassifier(oob_score=True, random_state=SEED, n_jobs=-1))
param_grid = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [4, 6, 8, None],
    "clf__max_features": ["sqrt", "log2", None],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                    scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_                      # refit on the whole training split
print("best parameters:", grid.best_params_)
print(f"best 5-fold CV F1: {grid.best_score_:.4f}")
print(f"OOB score (accuracy on out-of-bag training rows): {best_rf.named_steps['clf'].oob_score_:.4f}")
"""),
code("""
cvres = pd.DataFrame(grid.cv_results_)
cvres[["param_clf__n_estimators", "param_clf__max_depth", "param_clf__max_features",
       "mean_test_score", "rank_test_score"]].sort_values("rank_test_score").head(8).round(4)
"""),
code("""
results.loc["Random Forest (tuned)"] = pd.Series(evaluate(best_rf)).round(4)
results
"""),
md("""
**Tuning result.**
- **Search:** `GridSearchCV` tried 3 x 4 x 3 = 36 combinations with 5-fold stratified CV,
  scored by F1, all on the training split.
- **Best parameters:** **`n_estimators=400`, `max_depth=8`, `max_features=None`**.
- **Scores:** best mean CV F1 **0.772**. The refitted best forest, built with
  `RandomForestClassifier(oob_score=True, ...)`, reports an **OOB score of 0.833**: accuracy
  on the training rows each tree didn't see in its bootstrap sample, so a free internal
  validation estimate.
- **Why these parameters won:** capping depth at 8 stops the trees from memorizing the data,
  unlike the default forest's unlimited depth. `max_features=None` lets every split consider
  all features, which helps when one feature, `sex`, dominates.
- **Test set:** the tuned forest improves to **accuracy 0.831, precision 0.828, F1 0.762,
  AUC 0.841**, beating the default forest on every metric except recall, where they tie. Its
  test accuracy (0.831) also closely matches the OOB estimate (0.833), a sign it is not
  overfitting.
"""),
md("""
## Task 13 - Regression side-task: predict `fare` with multivariate linear regression

Features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `embarked`. `survived` is left out: it's an
outcome of the voyage, not something that could set the ticket price. The train/test rows are
the same as in the classification split, and the same fit-on-train preprocessor is used.
"""),
code("""
REG_NUMERIC = ["pclass", "age", "sibsp", "parch"]
reg_prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), REG_NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CATEGORICAL),
])
reg = Pipeline([("prep", reg_prep), ("lr", LinearRegression())])

Xr_train, Xr_test = X_train.drop(columns="fare"), X_test.drop(columns="fare")
fr_train, fr_test = X_train["fare"], X_test["fare"]
reg.fit(Xr_train, fr_train)
fare_pred = reg.predict(Xr_test)

n, p = len(fr_test), len(reg.named_steps["prep"].get_feature_names_out())
r2 = r2_score(fr_test, fare_pred)
reg_metrics = pd.Series({
    "MSE": mean_squared_error(fr_test, fare_pred),
    "MAE": mean_absolute_error(fr_test, fare_pred),
    "RMSE": np.sqrt(mean_squared_error(fr_test, fare_pred)),
    "R2": r2,
    "Adjusted R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
}, name="Linear Regression (fare)").round(4)
print(f"n = {n} test rows, p = {p} predictors after encoding")
reg_metrics
"""),
code("""
resid = fr_test - fare_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(fare_pred, resid, alpha=0.6, color="#4C72B0")
axes[0].axhline(0, color="k", ls="--")
axes[0].set(title="Residuals vs predicted fare", xlabel="predicted fare", ylabel="residual (actual - predicted)")
sns.boxplot(x=X_test["pclass"], y=resid, ax=axes[1], color="#DD8452")
axes[1].axhline(0, color="k", ls="--")
axes[1].set(title="Residual spread by passenger class", ylabel="residual")
fig.tight_layout()
save(fig, "12_regression_residuals.png")
plt.show()

spread = pd.DataFrame({"pred": fare_pred, "abs_resid": resid.abs().values})
spread["pred_band"] = pd.qcut(spread["pred"], 3, labels=["low", "mid", "high"])
print("residual std by predicted-fare band:")
print(spread.groupby("pred_band", observed=True)["abs_resid"].agg(["mean", "std"]).round(2).to_string())
print("Spearman corr(|residual|, predicted):", round(spread["abs_resid"].corr(spread["pred"], method="spearman"), 3))
"""),
md("""
**Regression results (test set, n = 178, p = 7 predictors):**

| Metric | Value |
|---|---:|
| MAE | 19.65 |
| MSE | 1702.6149 |
| RMSE | 41.26 |
| R² | 0.3474 |
| Adjusted R² | 0.3205 |

The model explains about a third of the variance in fare. It captures the class structure but
little else.

**Heteroscedasticity: yes, clearly present.** The residual plot fans out as the predicted fare
grows. The mean absolute residual rises from **9.6** in the lowest predicted-fare band to
**35.2** in the highest, and the residual standard deviation jumps from 7.3 to **59.4**. The
Spearman correlation between |residual| and the predicted value is **+0.44**, so the spread is
not random. The by-class box plot shows the same thing: 1st-class residuals span hundreds of
pounds, while 3rd-class residuals are tightly bunched.

The cause is that fare is extremely right-skewed and varies most within 1st class (different
cabins, suites, and group tickets). A linear model assumes constant error variance, so its
standard errors and intervals would be unreliable here. Modeling `log(fare)` would be the
natural fix.
"""),
md("""
## Task 14 - Model comparison table
Classification metrics (all on a 0-1 scale) and regression metrics (MAE/RMSE in fare units,MSE in fare units squared,
R-squared unitless) are **different metric groups on different scales**. They're kept in
separate column groups, and "-" marks a metric that doesn't apply to that model type.
"""),
code("""
cls = results.copy()
cls.columns = pd.MultiIndex.from_product([["Classification (target: survived, 0-1 scale)"],
                                          ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]])
rg = reg_metrics.to_frame().T
rg.columns = pd.MultiIndex.from_product([["Regression (target: fare)"],
                                         ["MAE","MSE","RMSE", "R2", "Adjusted R2"]])
final_table = pd.concat([cls, rg]).astype(object).where(lambda t: t.notna(), "-")
final_table
"""),
code("print(final_table.to_markdown())"),
md("""
**Recommendation.** I'd deploy the **tuned Random Forest**
(`n_estimators=400, max_depth=8, max_features=None`).
- **Accuracy, precision and F1:** it has the best test accuracy (**0.831**), precision
  (**0.828**) and F1 (**0.762**) of all the classifiers. The next best F1 is 0.734, from
  Logistic Regression.
- **Recall:** it ties for best recall (0.706).
- **Reliability:** its OOB score (0.833) closely matches its test accuracy, and its
  hyperparameters were chosen by cross-validation on the training data alone, so the test
  result is an honest estimate.
- **Trade-off:** Logistic Regression has a higher AUC (0.861 vs 0.841) and is easier to
  explain. It would be the better pick if calibrated probabilities or coefficient-level
  explanations mattered more than hard survived/died decisions.

The regression row sits in its own metric group. MAE/RMSE are in fare units,MSE in fare units squared and R² is a
share of variance explained, so none of those numbers can be compared with the
classification scores.
"""),
md("""
## Task 15 - Save the complete fitted pipeline and reload it on raw input
"""),
code("""
deploy_name = "Random Forest (tuned)"
deploy_pipe = best_rf              # Pipeline(ColumnTransformer -> RandomForestClassifier), already fitted

joblib.dump(deploy_pipe, MODELS / "titanic_best_pipeline.joblib")
print(f"saved {deploy_name}: {MODELS / 'titanic_best_pipeline.joblib'}")
print(deploy_pipe)
"""),
code("""
reloaded = joblib.load(MODELS / "titanic_best_pipeline.joblib")

# brand-new, raw, unpreprocessed passengers - including missing age / embarked
new_passengers = pd.DataFrame([
    {"pclass": 1, "age": 38,     "sibsp": 1, "parch": 0, "fare": 71.28, "sex": "female", "embarked": "C"},
    {"pclass": 3, "age": 22,     "sibsp": 0, "parch": 0, "fare": 7.25,  "sex": "male",   "embarked": "S"},
    {"pclass": 2, "age": np.nan, "sibsp": 0, "parch": 2, "fare": 26.0,  "sex": "female", "embarked": np.nan},
    {"pclass": 3, "age": 4,      "sibsp": 3, "parch": 1, "fare": 27.9,  "sex": "male",   "embarked": "Q"},
])
out = new_passengers.assign(
    predicted=reloaded.predict(new_passengers),
    p_survive=reloaded.predict_proba(new_passengers)[:, 1].round(3))
print(out.to_string(index=False))

# the reloaded object reproduces the in-memory pipeline exactly on the raw test set
assert (reloaded.predict(X_test) == deploy_pipe.predict(X_test)).all()
assert np.allclose(reloaded.predict_proba(X_test), deploy_pipe.predict_proba(X_test))
print("\\nReloaded pipeline matches the in-memory pipeline on all", len(X_test), "raw test rows;",
      f"test accuracy = {accuracy_score(y_test, reloaded.predict(X_test)):.4f}")
"""),
]


if __name__ == "__main__":
    for name, cells in [("01_eda.ipynb", EDA), ("02_modeling.ipynb", MODELING)]:
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, HERE / name)
        print(f"wrote {name} ({len(cells)} cells)")
