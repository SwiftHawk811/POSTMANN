# POSTMANN

Decision trees and random forests implemented from scratch in NumPy, built up in four stages: an unrestricted tree that overfits, a regularized tree that doesn't, a sanity check against scikit-learn, and a bagged ensemble on top.

No ML library is used for any of the learning logic. `scikit-learn` appears only to generate synthetic datasets and to provide a reference baseline.

## Layout

```
postman/
├── main/                          # the library
│   ├── __init__.py
│   ├── decisionTree.py            # unrestricted tree (grows until pure)
│   ├── regularized_decisionTree.py# depth / min-samples constrained tree
│   └── randomForest.py            # bagged ensemble of regularized trees
└── demo/                          # runnable experiments
    ├── __init__.py
    ├── overfitting.py             # stage 1: show the failure mode
    ├── regularisedTreeAccuracy.py # stage 2: constrain the tree
    ├── Scikit-comparison.py       # stage 3: reference baseline
    └── RandomForestComparison.py  # stage 4: ensemble vs single tree
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn
```

Requires Python 3.10+.

## Running

All demos are run as modules from the `postman/` directory, not as loose scripts:

```bash
cd postman
python -m demo.overfitting
python -m demo.regularisedTreeAccuracy
python -m demo.Scikit-comparison
python -m demo.RandomForestComparison
```

Running `python demo/overfitting.py` directly will fail — `main` is a package, and a direct script run doesn't put the project root on the import path.

## Results

Stages 1–3 use 150 samples, 10 features, 15% label noise, `random_state=42`, 75/25 split at `seed=1`. These are deterministic and reproduce exactly.

| Model | Train | Test |
|---|---|---|
| Unrestricted tree | 100.00% | 68.42% |
| Regularized tree (`max_depth=3`, `min_samples_split=8`) | 85.71% | 78.95% |
| scikit-learn `DecisionTreeClassifier` (same hyperparameters) | 85.71% | 68.42% |

Stage 4 uses a separate dataset — 200 samples, 8 features, 10% label noise — and is **not** seeded, so the forest figures move a little between runs.

| Model | Train | Test |
|---|---|---|
| Random forest (200 trees, `max_depth=4`) | ~88–91% | ~82–86% |
| Regularized tree (same data) | 82.67% | 78.00% |

The headline: memorizing the training set costs you ~31 points of test accuracy, constraining the tree buys most of it back, and averaging many decorrelated trees buys the rest.

Full analysis in [WRITEUP.md](WRITEUP.md).

## API

```python
from main.regularized_decisionTree import build_tree, predict
from main.randomForest import RandomForestClassifier

tree = build_tree(X_train, y_train, max_depth=3, min_samples_split=8)
y_pred = predict(tree, X_test)

forest = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=12)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
```

`max_features` accepts `"sqrt"` (default), `"log2"`, an `int` count, or a `float` fraction.

## Known limitations

- The forest does not seed its bootstrap sampling, so results are not reproducible run to run.
- `decisionTree.py` and `regularized_decisionTree.py` each carry their own copy of `gini`, `best_split`, `candidate_thresholds` and `predict`. The two differ only in stopping rules.
- Splits are evaluated on at most 20 quantile-spaced thresholds per feature, so very fine splits can be missed.
- Continuous features only; no categorical or missing-value handling.
- No pruning, no `class_weight`, no probability outputs.
