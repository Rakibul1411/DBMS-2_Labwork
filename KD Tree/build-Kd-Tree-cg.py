import pandas as pd
from scipy.spatial import KDTree

# Load the dataset
data = pd.read_csv('IRIS.csv')

# Extract numerical features for KDTree construction
numerical_features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Build the KD-tree
kd_tree = KDTree(numerical_features)

# Example query: Find the nearest neighbor of a given point
query_point = [5.0, 3.5, 1.4, 0.2]  # Example point
_, nearest_neighbor_index = kd_tree.query(query_point)

# Retrieve the nearest neighbor data
nearest_neighbor = numerical_features[nearest_neighbor_index]

# Output the results
print("Query Point:", query_point)
print("Nearest Neighbor:", nearest_neighbor)
