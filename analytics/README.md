# Module 2 - Analytics Pipeline (`/analytics`)

One cohesive pipeline on the Titanic dataset, split into two ordered notebooks that share one
committed CSV:

| Step | File | What it does |
|---|---|---|
| 1 | [`01_eda.ipynb`](01_eda.ipynb) | the **single** `sns.load_dataset('titanic')` call; saves `titanic.csv` immediately; profiling, missing-value handling, univariate and bivariate analysis, 5-chart data story, z-score check |
| 2 | [`02_modeling.ipynb`](02_modeling.ipynb) | reads the **same** `titanic.csv` (never reloads from seaborn); stratified split, `ColumnTransformer` pipeline, 3 classifiers, evaluation, imbalance comparison, `GridSearchCV` + OOB, fare regression, comparison table, `joblib` save and reload |
| - | `titanic.csv` | committed offline fallback (891 rows, raw, as loaded) |
| - | `models/titanic_best_pipeline.joblib` | the complete fitted pipeline (preprocessing + tuned Random Forest) |
| - | `figures/*.png` | charts saved by the notebooks (supporting artifacts only; every chart is interpreted in text) |
| - | [`build_notebooks.py`](build_notebooks.py) | the notebook source as plain Python, which regenerates both `.ipynb` files |

```bash
# from the repository root
jupyter nbconvert --to notebook --execute --inplace analytics/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace analytics/02_modeling.ipynb
```

Both notebooks are committed with their outputs. If `sns.load_dataset` can't reach the
internet, `01_eda.ipynb` falls back to `pd.read_csv("titanic.csv")`. All randomness is seeded
(`random_state=42`), so re-running reproduces every number below.

The full written interpretations live next to each output in the notebooks' Markdown cells.
The key results are summarized here.

---

## Part A - Profiling, cleaning, data story

### 1. Profile

891 rows x 15 columns. The target is imbalanced: **549 died (61.62%) and 342 survived
(38.38%)**.

| Column | Missing | % missing |
|---|---|---|
| `deck` | 688 | **77.22%** |
| `age` | 177 | **19.87%** |
| `embarked` | 2 | **0.22%** |
| `embark_town` | 2 | **0.22%** |

### 2. Missing-value handling (rule: < 5% drop rows, 5-30% impute, > 30% drop column or "missing" category)

| Column | % measured | Rule band | Action and justification |
|---|---|---|---|
| `embarked`, `embark_town` | 0.22% | < 5% | **Drop the 2 rows.** Negligible loss, and a guessed port would be fabricated. |
| `age` | 19.87% | 5-30% | **Impute the median within (`pclass`, `sex`) groups.** Age is skewed, so the median is safer than the mean. Age also differs by group (median 40 for 1st-class men vs 21.5 for 3rd-class women), so one global median would blur it. An `age_was_missing` flag is kept. |
| `deck` | 77.22% | > 30% | **Encode as its own category, `"Unknown"`.** Imputing 77% of values would be invention, but missingness is informative: 66.7% of passengers with a recorded deck survived vs 29.9% of "Unknown" (decks were recorded mostly for 1st class). Keeping the category preserves that signal without making up data. |

Result: 889 rows, 0 missing values.

### 3. Univariate: `age` and `fare`

Histograms and box plots for both columns are in `figures/01_univariate_age_fare.png`.

| | Q1 | Q3 | IQR | fences | **IQR outliers** |
|---|---|---|---|---|---|
| `fare` | 7.90 | 31.00 | 23.10 | [-26.76, 65.66] | **114** (all high) |
| `age` (cleaned) | 21.50 | 36.00 | 14.50 | [-0.25, 57.75] | **32** (all high) |
| `age` (raw, non-missing) | 20.12 | 38.00 | 17.88 | [-6.69, 64.81] | 11 |

Imputation stacks 177 values near the middle, which narrows the IQR, so the cleaned `age`
column flags more outliers than the raw one.

**`fare`: mean 32.10, median 14.45, mode 8.05.** The ordering **mode < median < mean**
means the distribution is **right-skewed** (skewness +4.80). Most tickets were cheap, and a
long tail of 1st-class fares up to 512.33 drags the mean upward.

### 4. Bivariate

Survival rates computed with boolean masks (`&`, `|`):

| Group | Survival rate |
|---|---|
| (a) female / male | **74.0%** / **18.9%** |
| (b) 1st / 2nd / 3rd class | **62.6%** / **47.3%** / **24.2%** |
| (c) female: 1st / 2nd / 3rd | **96.7%** / **92.1%** / **50.0%** |
| (c) male: 1st / 2nd / 3rd | **36.9%** / **15.7%** / **13.5%** |
| women or children under 16 (via `\|`) / adult men | 71.6% / 16.4% |

**Correlation matrix** on exactly `survived, pclass, age, sibsp, parch, fare`. `adult_male`
and `alone` are excluded as derived flags. The heatmap is `figures/02_correlation_heatmap.png`.
Ranked by |r| over the off-diagonal pairs:

1. **`pclass` vs `fare`: r = -0.548.** Cheaper classes (higher numbers) paid lower fares, so
   the ticket price is largely the class. The two columns carry overlapping
   wealth/status information.
2. **`sibsp` vs `parch`: r = +0.415.** People with siblings or spouses aboard also tended to
   have parents or children aboard: families travelled together, and both columns measure
   family size.

(Third is `pclass` vs `age` at -0.411. It is slightly inflated by the group-median imputation;
the raw value is -0.366.)

### 5. Multivariate data story: who survived, and why?

1. **Survival by class and sex** (`03_story_class_sex.png`). Sex dominates. Women out-survived
   men in every class (96.7% vs 36.9% in 1st, 50.0% vs 13.5% in 3rd). Class comes second:
   rates fall from 1st to 3rd for both sexes, so a 1st-class woman was about 7 times as likely
   to survive as a 3rd-class man. This is consistent with a "women and children first"
   evacuation where access favoured the upper classes.
2. **Survival by age band, split by sex** (`04_story_age_sex.png`). The "children first" rule
   shows up among males: boys under 16 survived at 52.5% vs 17.4% for adult men. Among women,
   survival is high at almost every age, so age adds little beyond sex. Only raw (non-imputed)
   ages are plotted, so imputation creates no artificial spike.
3. **Fare vs age, coloured by survival** (`05_story_fare_age.png`, log scale). Survivors cluster
   at higher fares, where 1st-class markers dominate. By fare quartile, survival climbs from
   19.7% (cheapest) to 57.7% (dearest). Fare acts as a proxy for class and cabin position
   close to the lifeboats.
4. **Survival by family size** (`06_story_family_size.png`). The effect is non-linear. Solo
   travellers survived at 30.1%, families of 2-4 at 55-72%, and families of 5 or more at only
   0-33% (mostly large 3rd-class households). That explains why `sibsp` and `parch` matter even
   though their linear correlation with `survived` is near zero.
5. **Survival by port and class** (`07_story_port_class.png`). Cherbourg's high overall rate
   (55.4% vs 33.7% for Southampton) is mostly class mix: 85 of its 168 passengers were in 1st
   class. Within a class the gap narrows (1st class: 69.4% vs 58.3%), and Queenstown's
   1st/2nd-class cells hold only 2 and 3 people. The port is a stand-in for class, not a cause
   in itself.

**The argument.** Survival was decided mainly by **sex**, then **class and wealth** (fare),
then **childhood** and **family size**. The port of embarkation mostly reflects each port's
class mix. These are the features the models go on to use, and the decision tree picks the
same ordering by itself.

### 6. z-score check (EDA only)

| | mean before | std before | mean after | std after |
|---|---|---|---|---|
| `age` | 29.07 | 13.27 | **0.0000** | **1.0000** |
| `fare` | 32.10 | 49.70 | **0.0000** | **1.0000** |

The overlaid histograms (`08_zscore_before_after.png`) show the shape is unchanged: `fare` is
still right-skewed. This check doesn't feed the model, which does its own train-only scaling.

---

## Part B - Predictive modeling (continues from the same `titanic.csv`)

`02_modeling.ipynb` re-applies Task 2's structural decisions: drop the 2 rows with no
`embarked`, and leave `deck` out. It does **not** reuse the full-data age imputation, since
that would leak test information. Features are `pclass, age, sibsp, parch, fare, sex,
embarked`. `alive` is excluded because it is the target restated as text, which would be pure
leakage. `class`, `who`, `embark_town`, `adult_male` and `alone` are excluded as duplicates or
derived columns.

### 7. Stratified split

80/20 with `stratify=y` and `random_state=42`. The classes are 61.75 / 38.25 (549 : 340). Without
stratification, a 178-row test set could by chance hold noticeably more or fewer survivors,
distorting every metric for the minority class. With it, the survivor share is **38.26% in
train and 38.20% in test**. The split happens before any preprocessing.

### 8. Preprocessing (fit on train only)

`ColumnTransformer` inside a `Pipeline`, so `fit` only ever sees training rows and the test set
is only transformed:

- numeric `pclass, age, sibsp, parch, fare`: `SimpleImputer(median)` -> `StandardScaler`
- categorical `sex, embarked`: `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")`

For example, the fitted age median (28.0) and scaler means (such as `fare` 31.86) come from
the 711 training rows. The notebook prints them.

### 9-10. Three classifiers on the identical split

Decision Tree: `max_depth=4, min_samples_leaf=5`, plotted with `plot_tree` using named
features and the classes `Died`/`Survived` (`09_decision_tree.png`). Its root split is
`sex_female` (63% of importance), then class and age (boys of about 3.5 or younger).

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 97 | 13 | 21 | 47 | 0.8090 | 0.7833 | 0.6912 | 0.7344 | **0.8610** |
| Decision Tree | 97 | 13 | 23 | 45 | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8510 |
| Random Forest (default) | 95 | 15 | 20 | 48 | 0.8034 | 0.7619 | 0.7059 | 0.7328 | 0.8237 |

Confusion matrices: `10_confusion_matrices.png`. ROC curves: `11_roc_curves.png`.

- **Logistic Regression** ranks survivors best: highest AUC.
- **Default Random Forest** catches one more survivor but makes more false alarms. Its
  unlimited depth overfits, which costs AUC.
- **All three models** miss about 30% of survivors, so recall is the weak spot.

### 11. Imbalance handling (Logistic Regression)

Training balance: 439 died vs 272 survived (1.61 : 1). SMOTE sits inside an `imblearn`
`Pipeline` after preprocessing, so it resamples **only the training fold** during `fit`
(to 439 : 439). Test rows are never resampled.

| Variant | Precision | Recall | F1 | 5-fold CV recall | 5-fold CV F1 |
|---|---|---|---|---|---|
| (a) baseline | **0.7833** | 0.6912 | 0.7344 | 0.6910 | 0.7100 |
| (b) `class_weight='balanced'` | 0.7183 | **0.7500** | 0.7338 | **0.7609** | 0.7269 |
| (c) SMOTE (train only) | 0.7353 | 0.7353 | **0.7353** | 0.7535 | **0.7278** |

**Conclusion.**
- **Trade-off:** both strategies swap precision for recall on the minority class, as expected.
- **Test set:** F1 is essentially tied (0.734-0.735), so one 178-row test set can't separate
  the variants.
- **Cross-validation:** both clearly beat the baseline on CV F1 (0.710 -> 0.727/0.728) and
  CV recall (0.691 -> 0.761/0.754).
- **Winner:** **`class_weight='balanced'`**. It gives the best recall on both test and CV,
  effectively ties SMOTE on F1, and is simpler, because it reweights the loss instead of
  inventing synthetic passengers.
- **Size of the effect:** the gain is modest because the imbalance is mild.

### 12. GridSearchCV + OOB (Random Forest)

- **Search:** `RandomForestClassifier(oob_score=True, random_state=42)` in the same pipeline;
  grid `n_estimators` {100, 200, 400} x `max_depth` {4, 6, 8, None} x `max_features`
  {sqrt, log2, None}; 5-fold stratified CV, F1 scoring, on the training split only.
- **Best parameters:** **`n_estimators=400, max_depth=8, max_features=None`**.
- **Best CV F1:** 0.7722.
- **OOB score:** **0.8326**, the accuracy on each tree's out-of-bag training rows.
- **Test set:** accuracy 0.8315, precision 0.8276, recall 0.7059, F1 0.7619, AUC 0.8406. That
  beats or ties the default forest on every metric, and test accuracy closely matches the OOB
  estimate.

### 13. Regression side-task: predict `fare`

Linear regression on `pclass, sex, age, sibsp, parch, embarked`, using the same train/test
rows and a fit-on-train preprocessor. `survived` is excluded because it is an outcome, not a
cause of the ticket price.

|| Metric | Value |
|---|---:|
| MAE | 19.65 |
| MSE | 1702.6149 |
| RMSE | 41.26 |
| R² | 0.3474 |
| Adjusted R² | 0.3205 |

| Metric | Value |
|---|---:|
| MAE | 19.65 |
| MSE | 1702.6149 |
| RMSE | 41.26 |
| R² | 0.3474 |
| Adjusted R² | 0.3205 |predicted fare grows:
- the mean |residual| rises from 9.6 in the lowest predicted-fare band to 35.2 in the highest;
- the residual std rises from 7.3 to 59.4;
- Spearman corr(|residual|, predicted) = +0.44;
- 1st-class residuals span hundreds of pounds, while 3rd-class residuals are tightly bunched.

The spread is not random, so the constant-variance assumption fails. Fare's heavy right skew
is the cause, and modeling `log(fare)` would be the natural fix.

### 14. Model comparison table

Classification and regression metrics are separate metric groups on different scales. They
are not comparable with each other, and "-" means "not applicable".

| Model | **Classification:** Accuracy | Precision | Recall | F1 | ROC AUC | **Regression:** MAE | MSE | RMSE | R² | Adj. R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | **0.8610** | - | - | - | - | - |
| Decision Tree | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8510 | - | - | - | - | - |
| Random Forest (default) | 0.8034 | 0.7619 | 0.7059 | 0.7328 | 0.8237 | - | - | - | - | - |
| **Random Forest (tuned)** | **0.8315** | **0.8276** | **0.7059** | **0.7619** | 0.8406 | - | - | - | - | - |
| Linear Regression (target `fare`) | - | - | - | - | - | 19.65 | 1702.6149 | 41.26 | 0.3474 | 0.3205 |
The classification columns are on a 0-1 scale (higher is better). MAE and RMSE are in fare
units, MSE is in fare units squared, and R² is the share of fare variance explained.

**Recommendation.** I'd deploy the **tuned Random Forest**.
- **Scores:** it has the best test accuracy (0.8315), precision (0.8276) and F1 (0.7619) of
  every classifier. The next-best F1 is Logistic Regression's 0.7344. It also ties for the
  best recall (0.7059).
- **Reliability:** its OOB score (0.8326) matches its test accuracy, and its hyperparameters
  were chosen by cross-validation on training data only, so the test estimate is honest.
- **Trade-off:** Logistic Regression's higher AUC (0.8610 vs 0.8406) and simple coefficients
  would make it the better choice if calibrated probabilities or explainability mattered more
  than hard survived/died decisions.

### 15. Saved pipeline

`joblib.dump(deploy_pipe, "models/titanic_best_pipeline.joblib")` saves the **whole fitted
`Pipeline`**: the `ColumnTransformer` (imputers, one-hot encoder, scaler) plus the tuned
`RandomForestClassifier`. It is not the bare estimator.

The last cell calls `joblib.load` and predicts on four brand-new **raw** passengers, including
one with missing `age` and `embarked`, which the embedded imputers handle. For example, the
1st-class woman gets p = 0.96 and the 3rd-class man p = 0.05. The cell then asserts the
reloaded pipeline's predictions and probabilities match the in-memory pipeline on all 178 raw
test rows (test accuracy 0.8315).
