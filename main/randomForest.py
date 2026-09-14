import numpy as np
from regularized_decisionTree import build_tree, predict

class RandomForestClassifier:
    def __init__(self, n_estimators = 15, max_depth = 3, min_samples_split = 8, max_features = "sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _get_feature_subset_size(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        return n_features

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        feature_subset_size = self._get_feature_subset_size(n_features)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            tree = build_tree(
                X_boot, y_boot,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_subset_size=feature_subset_size
            )
            self.trees.append(tree)
        return self

    def predict(self, X):
        tree_preds = np.array([predict(tree, X) for tree in self.trees])
        
        final_preds = []
        for i in range(X.shape[0]):
            sample_votes = tree_preds[:, i]
            vals, counts = np.unique(sample_votes, return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
            
        return np.array(final_preds)