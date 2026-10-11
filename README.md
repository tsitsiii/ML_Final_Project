# My ML Final Project!
## Short Description

A public agency wants to use survey data to flag households likely to be in a higher-income bracket (>50K), so that a benefit or outreach programme can be targeted. This project builds two models for the same classification problem, compares them fairly, and asks whether the neural network's extra complexity is worthwhile:

- **Logistic regression**: simple and transparent, so its decisions can be explained to the public.
- **Neural network** (`MLPClassifier`, two hidden layers of 32 and 16 units): more flexible, and potentially more accurate.

Because the classes are imbalanced (about 24% earn >50K), **F1 on the >50K class** is the main comparison metric. Confusion matrices are used to see whether the models make different kinds of errors. The analysis also covers fairness across sex, race and nationality, the different costs of false positives and false negatives, and the value of a transparent model.

## Setup

Requires Python 3.9+ and scikit-learn 1.1 or later. The project is meant to be run inside a virtual environment.

**1. Create the virtual environment** (skip this if you have already made it), from the project folder:

```bash
python -m venv venv
```

**2. Activate it:**

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

**3. Install the dependencies** (with the environment active):

```bash
pip install -r requirements.txt
```

## Method

1. **Data.** Adult Census Income from OpenML. `fnlwgt` (a survey sampling weight) and `education` (a duplicate of `education-num`) are dropped.
2. **Split.** 80% train / 20% test, stratified on the target, with a fixed random seed. The test set is not used until the final comparison.
3. **Preprocessing** (one `ColumnTransformer` inside each model pipeline, so each cross-validation fold learns its preprocessing only from its own training rows):
   - Numeric: median imputation, then standard scaling.
   - Categorical: most-frequent imputation, then one-hot encoding (categories with fewer than 20 training rows are grouped).
4. **Logistic regression**: 5-fold stratified cross-validation (F1, precision, recall, accuracy).
5. **Prediction.** A written prediction of the neural network's performance, recorded before running it.
6. **Neural network**: cross-validated on the same folds for a fair comparison.
7. **Test-set comparison.** Both models are fitted on the training set and evaluated on the same held-out test set. This includes confusion matrices, classification reports, fit times and a bootstrap interval for the F1 difference.
8. **Ethical analysis.**
   - Per-group selection rate, true positive rate, false positive rate and F1 for sex, race and nationality.
9. **Final interpretation and recommendation.**

## Results

### Cross-validation (5 folds, training set)

| Metric | Logistic regression | Neural network |
|---|---|---|
| F1 | 0.658 ± 0.008 | 0.672 ± 0.009 |
| Precision | 0.731 ± 0.016 | 0.737 ± 0.023 |
| Recall | 0.598 ± 0.005 | 0.619 ± 0.015 |
| Accuracy | 0.851 ± 0.005 | 0.856 ± 0.005 |

### Held-out test set

| Model | F1 | Precision | Recall | Accuracy | Fit time (s) |
|---|---|---|---|---|---|
| Logistic regression | 0.659 | 0.739 | 0.595 | 0.853 | 0.27 |
| Neural network | 0.682 | 0.739 | 0.633 | 0.859 | 1.52 |

### Confusion matrices (test set)

| Model | TP | FP | FN | TN |
|---|---|---|---|---|
| Logistic regression | 1390 | 491 | 948 | 6940 |
| Neural network | 1481 | 523 | 857 | 6908 |

### Fairness (test set, true positive rate / false positive rate)

| Group | Logistic regression | Neural network |
|---|---|---|
| Male | 0.608 / 0.095 | 0.648 / 0.101 |
| Female | 0.518 / 0.021 | 0.551 / 0.022 |
| White | 0.605 / 0.071 | 0.641 / 0.077 |
| Black | 0.450 / 0.023 | 0.550 / 0.021 |
| US | 0.599 / 0.067 | 0.640 / 0.072 |
| Non-US | 0.550 / 0.056 | 0.550 / 0.055 |

Groups with very few test rows (Amer-Indian-Eskimo, Other, Unknown nationality) are too small for reliable conclusions. The full per-group tables are in the notebook.

## Key findings

- The neural network scores slightly higher than logistic regression on every metric (test F1 0.682 vs 0.659). The gain comes almost entirely from recall: it finds 91 more high-income households at the cost of 32 more false positives.
- Both models make the same dominant error: they miss more high-income households (false negatives) than they wrongly flag.
- The improvement is small relative to the cross-validation noise and comes at a cost in training time, tuning effort, stability and explainability.
- Fairness is mixed. The sex gap is large and about the same in both models, the neural network narrows the White vs Black gap, and it widens the US vs non-US gap.
- **Recommendation:** logistic regression, because its transparency and auditability outweigh a gain of 1-2 F1 points. The conditions under which this recommendation would change are set out in Section 9 of the notebook.

## Limitations

- The neural network uses one reasonable configuration and was not tuned. Logistic regression was not tuned either.
- Fairness estimates come from a single test split without confidence intervals, and nationality is grouped coarsely into US, non-US and unknown.
- Fairness metrics here describe error rates only. Which fairness criterion matters most is a policy decision.
- The Adult data come from the 1994 US Census, so the results illustrate the method and do not describe current populations.



















