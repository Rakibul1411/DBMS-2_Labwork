import numpy as np
import pandas as pd
import heapq

class KDTreeNode:
    def __init__(self, point, label, axis, left=None, right=None):
        """
        Initializes a KDTreeNode.
        - point: The data point (coordinates).
        - label: The label associated with the point.
        - axis: The axis along which the split occurs.
        - left: The left subtree.
        - right: The right subtree.
        """
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self):
        self.root = None

    def _build_tree(self, points, labels, depth=0):
        """
        Recursively builds the KD-tree.
        - points: A numpy array of points.
        - labels: A numpy array of labels corresponding to the points.
        - depth: Current depth in the tree.
        """
        if len(points) == 0:
            return None

        k = points.shape[1]
        self.axis = depth % k  # Cycle through axes

        # Sort points along the axis
        sorted_idx = points[:, self.axis].argsort()
        points = points[sorted_idx]
        labels = labels[sorted_idx]

        # Select median as the splitting point
        median_idx = len(points) // 2

        # Create a KDTreeNode
        node = KDTreeNode(
            point = points[median_idx],
            label = labels[median_idx],
            axis = self.axis
        )

        # Recursively build left and right subtrees
        node.left = self._build_tree(points[:median_idx], labels[:median_idx], depth + 1)
        node.right = self._build_tree(points[median_idx + 1:], labels[median_idx + 1:], depth + 1)

        return node

    def build(self, points, labels):
        """
        Builds the KD-tree from the given points and labels.
        - points: A numpy array of points.
        - labels: A numpy array of labels corresponding to the points.
        """
        self.root = self._build_tree(points, labels)

    def _find_k_nearest_neighbors(self, node, target_point, k, depth=0, heap=None):
        """
        Recursively finds the k nearest neighbors using a max-heap.
        - node: Current node in the KD-tree.
        - target_point: The target point for which neighbors are sought.
        - k: Number of neighbors to find.
        - depth: Current depth in the tree.
        - heap: Max-heap to store the k nearest neighbors.
        """
        if node is None:
            return

        if heap is None:
            heap = []

        # Calculate distance to the current node
        current_distance = np.sqrt(np.sum((node.point - target_point) ** 2))

        # Push the current node into the heap if there's space, or replace the farthest neighbor
        if len(heap) < k:
            heapq.heappush(heap, (-current_distance, node))
        elif current_distance < -heap[0][0]:
            heapq.heappushpop(heap, (-current_distance, node))

        axis = node.axis
        next_node = node.left if target_point[axis] < node.point[axis] else node.right
        other_node = node.right if target_point[axis] < node.point[axis] else node.left

        # Search the next subtree
        self._find_k_nearest_neighbors(next_node, target_point, k, depth + 1, heap)

        # Check the other subtree if necessary
        if abs(target_point[axis] - node.point[axis]) < -heap[0][0]:
            self._find_k_nearest_neighbors(other_node, target_point, k, depth + 1, heap)

        return heap

    def find_k_nearest_neighbors(self, target_point, k):
        """
        Finds the k nearest neighbors to the target point.
        - target_point: The point to search neighbors for.
        - k: Number of nearest neighbors to find.
        """
        heap = self._find_k_nearest_neighbors(self.root, target_point, k)
        neighbors = [(node.point, node.label, -distance) for distance, node in sorted(heap, reverse=True)]
        return neighbors

def load_data_and_build_tree(dataset_path, label_column):
    """
    Loads the dataset and builds the KD-tree.
    - dataset_path: Path to the dataset (CSV file).
    - label_column: Name of the column containing labels.
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Dynamically extract feature columns
    feature_columns = [col for col in df.columns if col != label_column]
    features = df[feature_columns].values

    # Extract labels
    labels = df[label_column].values

    # Build the KD-tree
    kd_tree = KDTree()
    kd_tree.build(features, labels)

    return kd_tree

def main():
    """
    Main function to interact with the KDTree system.
    """
    # Configure dataset and label column
    dataset_path = 'IRIS.csv'  # Path to the dataset
    label_column = 'species'  # Label column name

    # Load data and build KD-tree
    kd_tree = load_data_and_build_tree(dataset_path, label_column)

    # User interaction loop
    while True:
        try:
            print("\nEnter the test point as values (e.g., 5.0 3.0 4.0 1.0):")
            user_input = input()
            target_point = np.array(list(map(float, user_input.split())))

            print("Enter the number of nearest neighbors you want to find (k):")
            k = int(input())

            # Find k nearest neighbors
            neighbors = kd_tree.find_k_nearest_neighbors(target_point, k)
            print(f"\nTest Point: {target_point}")
            print(f"{k} Nearest Neighbors:")

            for idx, (neighbor_point, neighbor_label, distance) in enumerate(neighbors, start=1):
                print(f"{idx}. Point: {neighbor_point}, Label: {neighbor_label}, Distance: {distance:.2f}")

        except Exception as e:
            print(f"Error: {e}")

        print("\nDo you want to continue? (yes/no)")
        if input().strip().lower() != 'yes':
            break

# Entry point
if __name__ == "__main__":
    main()
