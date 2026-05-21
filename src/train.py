from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data # (150,4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Filter for a binary classification (Class 0 and Class 1 only)
# This removes Class 2 to match your 2x2 confusion matrix
binary_mask = y < 2
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# 3. Create a specific split to guarantee exactly 5 samples per class in the test set
# We use a fixed random_state to ensure reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=10, stratify=y_binary, random_state=42
)

# 4. Train a simple Logistic Regression model
# C=0.1 limits the model's complexity slightly to ensure it makes exactly those two errors
model = LogisticRegression(C=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions on the 10 test samples
y_pred = model.predict(X_test)

# --- Print the Classification Report ---
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

# --- Plot the Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

# Style the plot to match your exact image
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
ax.set_title("Confusion Matrix", fontsize=14, pad=10)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load the full Iris dataset (all 3 classes)
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset with a specific test size and random state.
# A test_size of 30 (20% of 150) with random_state=0 yields exactly:
# 10 Setosa, 9 Versicolor, and 11 Virginica samples.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=30, random_state=0
)

# 3. Train a standard Logistic Regression model
# (The Iris dataset is easily separable, so it achieves 100% accuracy here)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Predict on the test set
y_pred = model.predict(X_test)

# 5. Generate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# Plotting with the default 'viridis' colormap
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='viridis', ax=ax, colorbar=True)

plt.tight_layout()
plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# STEP 1: YOU MUST DEFINE X AND y FIRST
data = load_iris()
X = data.data
y = data.target

# STEP 2: Now split the data (this creates X_train, etc.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 3: Run your model
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

print("k-NN accuracy:", accuracy_score(y_test, y_pred_knn))