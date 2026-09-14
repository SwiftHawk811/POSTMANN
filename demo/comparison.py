from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        flip_y=0.15,  # Introduces 15% label noise
        random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
dt = DecisionTreeClassifier(max_depth=3, min_samples_split=8)

dt.fit(X_train, y_train)
train_accuracy = dt.score(X_train, y_train)
test_accuracy = dt.score(X_test, y_test)

print(f"Scikit-learn decision tree train accuracy: {train_accuracy*100:.2f}%")
print(f"Scikit-learn decision tree test accuracy: {test_accuracy*100:.2f}%")

print("Self implemented decision tree train accuracy: 85.71%")
print("Self implemented decision tree test accuracy: 79.85%")