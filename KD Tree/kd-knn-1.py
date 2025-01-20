import pandas as pd
import heapq
from math import sqrt


class TreeNode:
    def __init__(self, point, axis):
        self.point = point  # Store the coordinates of the point
        self.axis = axis  # Axis along which the split occurs
        self.left = None  # Left child node
        self.right = None  # Right child node


class KDTree:
    def __init__(self, data):
        self.data = data  # Data for building the tree
        # print(self.data)
        self.root = None  # Root node of the KD-Tree

    def _construct(self, points, depth):
        if not len(points):
            return None

        k = len(points.columns)  # Number of dimensions
        axis = depth % k  # Current splitting axis
        sort_column = points.columns[axis]
        
        # print(sort_column)

        # Sort the points by the selected axis
        points = points.sort_values(by=[sort_column])
        # print(points)
        median_idx = len(points) // 2

        # Create a tree node with the median point
        node = TreeNode(
            point=points.iloc[median_idx].values.tolist(),
            axis=axis
        )

        # Recursively build the left and right subtrees
        node.left = self._construct(points.iloc[:median_idx], depth + 1)
        node.right = self._construct(points.iloc[median_idx + 1:], depth + 1)

        return node

    def build(self):
        self.root = self._construct(self.data, depth=0)

    def _calculate_distance(self, a, b):
        """Compute the Euclidean distance between two points."""
        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _search_knn(self, node, target, k, max_heap):
        if not node:
            return

        # Compute the distance to the current node
        dist = self._calculate_distance(target, node.point)

        # Maintain the k smallest distances in the max-heap
        heapq.heappush(max_heap, (-dist, node.point))
        if len(max_heap) > k:
            heapq.heappop(max_heap)

        axis = node.axis
        primary = node.left if target[axis] < node.point[axis] else node.right
        secondary = node.right if target[axis] < node.point[axis] else node.left

        # Recursively search the more promising branch first
        self._search_knn(primary, target, k, max_heap)

        # Check the other branch only if it may contain closer points
        if abs(target[axis] - node.point[axis]) < -max_heap[0][0]:
            self._search_knn(secondary, target, k, max_heap)

    def find_knn(self, target, k):
        """Locate the k-nearest neighbors of the target point."""
        max_heap = []
        self._search_knn(self.root, target, k, max_heap)
        return [point for _, point in sorted(max_heap, reverse=True)]


def load_dataset(filepath):
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(filepath)


def run_program():
    file_path = r"IRIS.csv"

    try:
        # Load and process the dataset
        iris_df = load_dataset(file_path)
        features = iris_df.drop(columns='species')  # Exclude the target column

        # Input target point and number of neighbors
        target_point = list(map(float, input("Enter the target point as space-separated values (e.g., 5.5 3.5 1.3 0.2): ").split()))
        
        print(target_point)
        
        k = int(input("Enter the number of neighbors (k): "))

        # Build and query the KD-Tree
        tree = KDTree(features)
        
        tree.build()
        
        neighbors = tree.find_knn(target_point, k)

        # Display the results
        print("k-Nearest Neighbors:")
        for neighbor in neighbors:
            print(neighbor)

    except FileNotFoundError:
        print("Error: File not found. Check the file path.")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")


if __name__ == "__main__":
    run_program()
