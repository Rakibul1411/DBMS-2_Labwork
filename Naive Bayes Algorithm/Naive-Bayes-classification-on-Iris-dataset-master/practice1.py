from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()

# Initialize the Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(iris.data, iris.target)

# Make predictions
predicted = model.predict(iris.data)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(iris.target, predicted))
print("\nClassification Report:\n", metrics.classification_report(iris.target, predicted))
print("\nConfusion Matrix:\n", metrics.confusion_matrix(iris.target, predicted))

# Visualize the target classes using matplotlib
plt.figure(figsize=(8, 6))
for target_class in np.unique(iris.target):
    plt.scatter(
        iris.data[iris.target == target_class, 2],  # Petal length
        iris.data[iris.target == target_class, 3],  # Petal width
        label=f"Class {target_class}"
    )

plt.title("Iris Dataset Target Classes Visualization")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.show()