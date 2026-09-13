import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main.regularized_decisionTree import build_tree, predict

from sklearn.datasets import make_classification

# 1. Generate a small, noisy synthetic dataset
X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        flip_y=0.15,  # Introduces 15% label noise
        random_state=42)

def train_test_split(X, y, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    train, test = idx[:cut], idx[cut:]
    return X[train], X[test], y[train], y[test]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)

regularized_tree = build_tree(X_train, y_train)
print(f"Regularized tree train accuracy: {accuracy(y_train, predict(regularized_tree, X_train))*100:.2f}%")
print(f"Regularized tree test accuracy: {accuracy(y_test, predict(regularized_tree, X_test))*100:.2f}%")