import numpy as np
from scipy.spatial import KDTree
import pandas as pd

# Load the IRIS dataset
data = pd.read_csv('IRIS.csv')

# Extract features (sepal_length, sepal_width, petal_length, petal_width)
features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values


# Build the kd-tree
kd_tree = KDTree(features)

# Example query: Find the nearest neighbors to a given point
query_point = [5.1, 3.5, 1.4, 0.2]  # Example query point (Iris-setosa)
distances, indices = kd_tree.query(query_point, k=3)  # Find 3 nearest neighbors

# Print the results
print("Query Point:", query_point)
print("Nearest Neighbors Indices:", indices)
print("Distances to Neighbors:", distances)
print("Nearest Neighbors Data:")
for idx in indices:
    print(features[idx], data.iloc[idx]['species'])