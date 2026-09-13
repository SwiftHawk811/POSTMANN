import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p ** 2)

def candidate_thresholds(column, max_candidates=20):
    vals = np.unique(column)
    if len(vals) <= max_candidates:
        return vals
    return np.unique(np.quantile(column, np.linspace(0.05, 0.95, max_candidates)))

def best_split(X, y, feature_subset=None):
    """Return (feature, threshold, gain) of the split that reduces Gini most."""
    n_samples, n_features = X.shape
    parent = gini(y)
    best_feature, best_threshold, best_score = None, None, parent

    features = range(n_features) if feature_subset is None else feature_subset
    for f in features:
        for t in candidate_thresholds(X[:, f]):
            left = X[:, f] <= t
            n_left = left.sum()
            if n_left == 0 or n_left == n_samples:
                continue
            score = (n_left * gini(y[left]) +
                     (n_samples - n_left) * gini(y[~left])) / n_samples
            if score < best_score:
                best_feature, best_threshold, best_score = f, t, score

    if best_feature is None:
        return None, None, 0.0
    return best_feature, best_threshold, parent - best_score

class Node:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def majority(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]


def build_unrestricted_tree(X, y, depth=0, feature_subset_size=None):
    if len(np.unique(y)) == 1 or len(y) <= 1:
        return Node(leaf=True, prediction=majority(y), n=len(y))
    subset = None
    if feature_subset_size is not None:
        subset = np.random.choice(X.shape[1], feature_subset_size, replace=False)
    f, t, gain = best_split(X, y, subset)
    if f is None:
        return Node(leaf=True, prediction=majority(y), n=len(y))
    mask = X[:, f] <= t
    return Node(leaf=False, feature=f, threshold=t, gain=gain, n=len(y),
                left=build_unrestricted_tree(X[mask], y[mask], depth + 1, feature_subset_size),
                right=build_unrestricted_tree(X[~mask], y[~mask], depth + 1, feature_subset_size))

def predict_one(node, x):
    while not node.leaf:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction


def predict(tree, X):
    return np.array([predict_one(tree, row) for row in X])