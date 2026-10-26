# Optimizing Model Prediction: Forest Fire Burn Area

Regression on the UCI Forest Fires dataset (517 fires, Montesinho park, Portugal), predicting burned area from weather and fire-weather indices.

Dataquest guided project, Data Scientist in Python path.

**The result is that burn area is not predictable from this data, and the notebook proves it rather than reporting a near-zero score as a success.** The brief was model optimization; the optimization was done properly and then tested against the possibility that there was nothing to optimize.

## Every model lost to predicting a constant

| Model | RMSE (log) | MAE (log) | R-squared |
|---|---|---|---|
| **Baseline: predict the training mean** | **1.4830** | 1.2028 | **-0.0006** |
| Lasso (alpha=0.05) | 1.4849 | 1.1904 | -0.0033 |
| ElasticNet (alpha=0.05) | 1.4987 | 1.1943 | -0.0219 |
| Random forest, 400 trees | 1.5087 | 1.2073 | -0.0356 |
| Ridge (alpha=10) | 1.5229 | 1.2021 | -0.0553 |
| Linear regression | 1.5277 | 1.2048 | -0.0619 |
| Gradient boosting | 1.6091 | 1.2575 | -0.1780 |
| Baseline: predict the training median | 1.6432 | 1.1334 | -0.2285 |

Lasso comes closest to the baseline precisely because `alpha=0.05` shrinks nearly every coefficient to zero, so it is approximating the mean predictor. That is not a model succeeding, it is a model correctly concluding there is nothing to fit.

Across 20 folds, `DummyRegressor` ranks first at -0.0066. Ridge ranges from **-1.5343 to +0.0514** across individual folds, so reporting the best fold would claim R-squared 0.05 and the worst would claim -1.53. The mean is the honest number and it is negative.

## The permutation test settles it

A negative R-squared is bad but not conclusive. Refitting Ridge on 40 deliberately shuffled targets establishes what pure noise earns:

| | 5-fold CV R-squared |
|---|---|
| **Real target** | **-0.0925** |
| Shuffled, mean | -0.1100 |
| Shuffled, std | 0.0401 |
| Shuffled, range | -0.2416 to -0.0366 |
| Shuffled, 95th percentile | -0.0574 |
| **Permutation p-value** | **0.30** |

Thirty per cent of random relabellings scored at least as well as the real data. There is no detectable relationship between these twelve predictors and burned area at n=517.

This is what makes the rest defensible. Without it, "my R-squared was about zero" is an admission. With it, "there is no signal here, and here is the evidence" is a conclusion.

## Why optimization could not have worked

**The target is zero-inflated, not skewed.** 247 of 517 fires (47.8%) burned exactly zero area. Skew 12.85, kurtosis 194.14. `log1p` reduces skew but leaves a point mass of half the data at exactly zero, which no monotonic transform can move. A squared-error regressor facing that minimises loss by predicting near the overall mean for everyone, which is what the better-regularised models did.

**The tail is a handful of events.** The single largest fire is 16.4% of all hectares burned; the top five are 38.1%. Predicting total area means predicting those five fires, and there are five.

**No predictor correlates with the target.** The strongest absolute correlation across Pearson-on-raw, Pearson-on-log and Spearman is `temp` at 0.098, which explains under 1% of variance.

**There is a mechanistic reason.** FFMC, DMC, DC and ISI describe how readily fuel ignites and spreads. How large a fire becomes depends on detection time, suppression response, crew availability, terrain and wind shifts during the burn, none of which is in the dataset. The features describe ignition conditions; the target measures the outcome of a firefighting operation.

## Every optimization decision was also irrelevant

| Imputer | CV R-squared |
|---|---|
| KNN (k=5) | -0.1737 |
| Constant (0) | -0.1761 |
| Mean | -0.1787 |
| Median | -0.1816 |

Spread between best and worst: 0.0079, against a fold-to-fold std above 0.34. All negative.

**On outliers the reflex was also wrong.** The IQR rule flags 63 fires (12.2%) as `area` outliers, and those 63 account for the overwhelming majority of all hectares burned. Removing them would delete the phenomenon from a dataset about the phenomenon, and would inflate the apparent score by making the target artificially easy. The IQR fence assumes approximate normality; kurtosis here is 194. They stayed in.

## Reframing found a little signal, but not much

| Task | Model | CV ROC AUC | std |
|---|---|---|---|
| Burned vs not (n=517) | Logistic regression | 0.5058 | 0.0248 |
| Burned vs not | Random forest | **0.5677** | 0.0517 |
| Large vs small, burned only (n=270) | Logistic regression | 0.5550 | 0.0408 |
| Large vs small, burned only | Random forest | **0.5775** | 0.0622 |

Chance is 0.5. Logistic regression on "did it burn anything" is exactly chance. The random forest reaches 0.568 and 0.578, weakly above chance but roughly one standard deviation from nothing.

Classification is better than regression here and still close to useless. If something had to be shipped from this dataset it would be a random forest classifier on whether a fire burns a measurable area, with the explicit caveat that it is barely better than a coin flip.

## On the feature importances

`temp` ranks first in both framings (0.179 regression, 0.136 classification), then `RH`, `DMC`, `DC`. These are reported for completeness and are **not** findings. An importance from a model that does not predict better than chance describes how that model allocated its splits, not what drives fire size. Publishing them as drivers of fire behaviour would be the most misleading thing here.

## Method

1. Load with `index_col=0` and `na_values='NA'` (232 injected missing cells across 8 columns).
2. Test whether the missingness relates to the target before choosing an imputer.
3. Characterise the target: zero fraction, skew, kurtosis, tail concentration.
4. Correlations against raw and log target, Pearson and Spearman.
5. Compare four imputation strategies on identical folds inside pipelines.
6. Audit outliers and argue explicitly for keeping them.
7. Establish mean and median baselines **before** fitting anything.
8. Sweep linear, Ridge, Lasso, ElasticNet, random forest and gradient boosting.
9. **Permutation test** against 40 shuffled targets.
10. Reframe as two classification problems and compare against chance.

## Limitations

- 517 rows is small, 270 once zeros are dropped. The correct claim is "no detectable signal at this sample size", not "no signal exists".
- The permutation test used Ridge only. A different model class could in principle find structure Ridge cannot, though the forests and boosting also scored negative.
- The missing values were injected for the exercise, so the missingness analysis is a methodological demonstration, not a finding about fire records.
- The zero-inflated structure invites a model built for it, which this notebook does not fit. A two-stage hurdle model or a Tweedie GLM would be the statistically appropriate choice; the weak AUC in section 9 suggests it would not rescue the problem, but that is an inference rather than a result.
- `area` carries no information about detection or suppression, the variables most likely to explain it, so the ceiling may be a data-collection limit rather than a modelling one.

## Stack

Python, pandas, NumPy, scikit-learn (Pipeline, SimpleImputer, KNNImputer, StandardScaler, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, DummyRegressor, RepeatedKFold, StratifiedKFold), matplotlib.

## Files

- `notebook.ipynb`: the full analysis with executed outputs and figures.