import numpy as np
import pandas as pd

class KDNode:
    def __init__(self, point, label, left=None, right=None):
        self.point = point
        self.label = label
        self.left = left
        self.right = right

class KDTree:
    def __init__(self):
        self.root = None
        
    def _build_tree(self, points, labels, depth=0):
        if len(points) == 0:
            return None
            
        # Get number of features (k)
        k = points.shape[1]
        
        # print(k)
        
        # Select axis based on depth
        axis = depth % k
        
        # Sort points and labels based on the selected axis
        sorted_idx = points[:, axis].argsort()
        points = points[sorted_idx]
        labels = labels[sorted_idx]
        
        # Get median index
        median_idx = len(points) // 2
        
        # Create node and construct subtrees
        node = KDNode(
            point=points[median_idx],
            label=labels[median_idx]
        )
        
        node.left = self._build_tree(points[:median_idx], labels[:median_idx], depth + 1)
        node.right = self._build_tree(points[median_idx + 1:], labels[median_idx + 1:], depth + 1)
        
        return node
    
    def build(self, points, labels):
        """Build KD-tree from points and labels"""
        self.root = self._build_tree(points, labels)
    
    def _find_nearest_neighbor(self, node, point, depth=0, best=None, best_dist=float('inf')):
        if node is None:
            return best, best_dist
            
        k = len(point)
        axis = depth % k
        
        # Calculate current distance
        current_dist = np.sqrt(np.sum((node.point - point) ** 2))
        
        # Update best if current node is closer
        if current_dist < best_dist:
            best = node
            best_dist = current_dist
        
        # Recursively search left or right subtree
        next_node = node.left if point[axis] < node.point[axis] else node.right
        other_node = node.right if point[axis] < node.point[axis] else node.left
        
        # Traverse down the tree
        best, best_dist = self._find_nearest_neighbor(next_node, point, depth + 1, best, best_dist)
        
        # Check if we need to search the other subtree
        if abs(point[axis] - node.point[axis]) < best_dist:
            best, best_dist = self._find_nearest_neighbor(other_node, point, depth + 1, best, best_dist)
        
        return best, best_dist
    
    def find_nearest_neighbor(self, point):
        """Find nearest neighbor to given point"""
        best_node, best_dist = self._find_nearest_neighbor(self.root, point)
        return best_node.point, best_node.label, best_dist

# Example usage
def load_and_build_kdtree():
    # Read the data
    df = pd.read_csv('IRIS.csv')
    
    # Extract features and labels
    features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    labels = df['species'].values
    
    # print(features)
    
    # print(labels)
    
    # Create and build KD-tree
    kdtree = KDTree()
    kdtree.build(features, labels)
    
    return kdtree

# Create the tree
kdtree = load_and_build_kdtree()

# Example: Find nearest neighbor for a test point
test_point = np.array([5.0, 3.0, 4.0, 1.0])
nearest_point, nearest_label, distance = kdtree.find_nearest_neighbor(test_point)

print(f"Test point: {test_point}")
print(f"Nearest neighbor point: {nearest_point}")
print(f"Nearest neighbor label: {nearest_label}")
print(f"Distance: {distance:.2f}")