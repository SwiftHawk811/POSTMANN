import numpy as np
from randomForest import RandomForestClassifier
from regularized_decisionTree import build_tree, predict
from sklearn.datasets import make_classification

def train_test_split(X, y, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    train, test = idx[:cut], idx[cut:]
    return X[train], X[test], y[train], y[test]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

# 1. Generate a small, noisy synthetic dataset
X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        n_classes=2,
        flip_y=0.1, 
        random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)

model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=12)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_acc = accuracy(y_train, y_train_pred)
test_acc = accuracy(y_test, y_test_pred)
print(f"Random forest train accuracy: {train_acc*100:.2f}%")
print(f"Random forest test accuracy: {test_acc*100:.2f}%")

regularized_tree = build_tree(X_train, y_train)
print(f"Regularized tree train accuracy: {accuracy(y_train, predict(regularized_tree, X_train))*100:.2f}%")
print(f"Regularized tree test accuracy: {accuracy(y_test, predict(regularized_tree, X_test))*100:.2f}%")
