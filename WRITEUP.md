# Writeup: from an overfitting tree to a random forest

## The question

A decision tree left to its own devices will keep splitting until every leaf is pure. On clean data that sounds ideal. On noisy data it is a trap: the tree ends up carving out a separate region for every mislabelled point, and those regions describe the noise, not the pattern.

This writeup walks that failure and the two standard fixes — regularization, then ensembling — measuring each one.

## Setup

All experiments use `sklearn.datasets.make_classification` for data and nothing else from scikit-learn except a reference baseline in stage 3.

Stages 1–3:

| Parameter | Value |
|---|---|
| `n_samples` | 150 |
| `n_features` | 10 (5 informative, 2 redundant) |
| `flip_y` | 0.15 |
| `random_state` | 42 |
| Split | 75 / 25, `seed=1` |

`flip_y=0.15` flips the label on roughly 15% of rows. Those rows are unlearnable by construction — any model that fits them is fitting a coin toss.

## How a split is chosen

Before the results, the one piece of machinery everything rests on.

Gini impurity measures how mixed a group of labels is:

$$G(y) = 1 - \sum_{k} p_k^2$$

Think of it as reaching into a bag of labelled marbles twice and asking how often you draw two different colours. All one colour → 0. Evenly mixed two classes → 0.5.

A candidate split is scored by the size-weighted impurity of the two halves it produces:

$$G_{\text{split}} = \frac{n_L}{n} G(y_L) + \frac{n_R}{n} G(y_R)$$

and the gain is $G(y_{\text{parent}}) - G_{\text{split}}$. `best_split` sweeps every feature and every candidate threshold and keeps the largest gain.

### Worked trace

Take 8 samples at a node, labels `[0,0,0,0,1,1,1,1]`, and one feature with values `[1,2,3,4,5,6,7,8]`.

**Parent impurity.** $p_0 = p_1 = 0.5$, so $G = 1 - (0.25 + 0.25) = 0.5$.

**Candidate $t = 4$.** Left gets values ≤ 4, labels `[0,0,0,0]`; right gets `[1,1,1,1]`.

- $G(y_L) = 1 - 1^2 = 0$
- $G(y_R) = 1 - 1^2 = 0$
- $G_{\text{split}} = \frac{4}{8}(0) + \frac{4}{8}(0) = 0$
- Gain $= 0.5 - 0 = 0.5$ — the maximum possible.

**Candidate $t = 2$.** Left `[0,0]`, right `[0,0,1,1,1,1]`.

- $G(y_L) = 0$
- $G(y_R) = 1 - \left(\left(\tfrac{2}{6}\right)^2 + \left(\tfrac{4}{6}\right)^2\right) = 1 - (0.111 + 0.444) = 0.444$
- $G_{\text{split}} = \frac{2}{8}(0) + \frac{6}{8}(0.444) = 0.333$
- Gain $= 0.5 - 0.333 = 0.167$

$t = 4$ wins. That is the entire decision rule, applied recursively.

One implementation detail: rather than test every unique value, `candidate_thresholds` caps the search at 20 quantile-spaced points per feature. This trades a small amount of split quality for a large speedup, and on continuous features the loss is usually negligible.

## Stage 1 — the unrestricted tree

`build_unrestricted_tree` has exactly two stopping conditions: the node is pure, or it holds one sample. Nothing else halts it.

```
Unrestricted tree train accuracy: 100.00%
Unrestricted tree test accuracy:   68.42%
```

A 31.6-point gap. The tree reproduced the training labels perfectly, including the ~15% that were flipped at random, and paid for it on data it hadn't seen. This is textbook high variance: the model is sensitive to which particular points happened to land in the training set.

The tell is that 100% train accuracy is *achievable at all* on data with 15% injected noise. A model that can hit 100% there is not learning structure — it is memorizing.

## Stage 2 — regularization

`build_tree` adds two stopping rules before any split is attempted:

```python
if (depth >= max_depth or len(y) < min_samples_split
        or len(np.unique(y)) == 1):
    return Node(leaf=True, prediction=majority(y), n=len(y))
```

- **`max_depth=3`** caps the tree at 8 leaves. With 112 training rows, each leaf averages ~14 samples — far too coarse to isolate individual noisy points.
- **`min_samples_split=8`** refuses to split any node holding fewer than 8 samples, on the grounds that a split justified by 3 points is not evidence.

Both rules work the same way: they make it structurally impossible for the tree to build a region small enough to fit one mislabelled row.

```
Regularized tree train accuracy: 85.71%
Regularized tree test accuracy:  78.95%
```

Train accuracy dropped 14 points. Test accuracy rose 10.5. The gap narrowed from 31.6 to 6.8.

Note that train accuracy of ~86% on data with 15% noise is roughly what a well-calibrated model *should* achieve. The ceiling is the noise floor, and the regularized tree sits just above it.

## Stage 3 — scikit-learn baseline

Running `DecisionTreeClassifier(max_depth=3, min_samples_split=8)` on the same generated data:

```
Scikit-learn decision tree train accuracy: 85.71%
Scikit-learn decision tree test accuracy:  68.42%
```

Train accuracy matches the from-scratch implementation exactly, which is the meaningful signal: the same hyperparameters produce the same fit quality, so the Gini criterion and the recursive splitting are implemented correctly.

The test figures differ because the two are not evaluated on the same test set. `Scikit-comparison.py` calls `sklearn.model_selection.train_test_split(..., random_state=42)`, while the from-scratch demos use their own `train_test_split(..., seed=1)`. Different permutations, different held-out rows, and with only 38 test samples a handful of reassigned points moves the number several percent.

**This comparison should be rerun on a shared split before the test-accuracy line is treated as evidence.** As it stands, only the train-accuracy match is a valid check. (`Scikit-comparison.py` also hardcodes the self-implemented figure as `79.85%`, where the actual value is `78.95%` — a transposed digit.)

## Stage 4 — the random forest

Regularization shrinks variance by weakening one model. Ensembling shrinks it by averaging many.

`RandomForestClassifier` builds `n_estimators` regularized trees, each on its own bootstrap resample, each split restricted to a random subset of features:

```python
indices = np.random.choice(n_samples, size=n_samples, replace=True)
tree = build_tree(X[indices], y[indices],
                  max_depth=self.max_depth,
                  min_samples_split=self.min_samples_split,
                  feature_subset_size=feature_subset_size)
```

Two sources of randomness, doing different jobs:

**Bootstrap sampling** draws `n` rows with replacement, so each tree sees roughly 63% of the unique training data and a different slice of the noise. The mislabelled rows that mislead one tree are absent from the next.

**Feature subsampling** (`max_features="sqrt"`, so 2 of 8 features here) is the less obvious half and the more important one. Without it, every tree would find the same dominant feature and split on it first, and the trees would end up near-identical. Averaging correlated models buys nothing. Forcing each split to choose among a random handful pushes the trees apart, and only *decorrelated* errors cancel.

Prediction is a majority vote across trees.

Stage 4 uses a different dataset — 200 samples, 8 features, `flip_y=0.1` — so these figures are not directly comparable to stages 1–3. Both models below were run on this same data:

```
Random forest    train ~88–91%   test ~82–86%
Regularized tree train  82.67%   test  78.00%
```

The forest gains roughly 4–8 points of test accuracy over the single tree, for ~200× the compute.

The range is not measurement noise — `np.random.choice` is called without a seed, so every run bootstraps differently. **Threading a `random_state` through `fit` would make this reproducible** and is the first thing to fix.

## Summary

| Stage | Model | Train | Test | Gap |
|---|---|---|---|---|
| 1 | Unrestricted tree | 100.00% | 68.42% | 31.58 |
| 2 | Regularized tree | 85.71% | 78.95% | 6.76 |
| 3 | scikit-learn (reference) | 85.71% | 68.42%\* | — |
| 4 | Random forest | ~90%† | ~84%† | ~6 |

\* different test split, see stage 3  †different dataset and unseeded, see stage 4

The progression makes one point three ways. Stage 1 shows that unconstrained capacity on noisy data buys memorization, not generalization. Stage 2 shows that hard structural limits recover most of the loss by making overfitting impossible rather than merely unattractive. Stage 4 shows that averaging decorrelated weak learners recovers more still, and that the decorrelation — not the averaging — is what makes it work.
