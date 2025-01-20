import pandas as pd
import heapq
from math import sqrt


class KDTreeNode:
    def __init__(self, point, axis):
        self.point = point  # Store the current point as a list
        self.axis = axis  # Axis along which the split occurs (0 for SepalLength, 1 for SepalWidth, etc.)
        self.left = None  # Pointer to the left subtree
        self.right = None  # Pointer to the right subtree


class KD_Tree:
    def __init__(self, data):
        self.data = data  # Input data as a pandas DataFrame
        self.root = None  # Root of the KD-Tree

    def _build(self, points, depth):
        if len(points) == 0:
            return None

        k = len(points.columns)  # Number of dimensions
        axis = depth % k  # Current axis
        column = points.columns[axis]

        # Sort points by the current axis and find the median
        points = points.sort_values(by=[column])
        median_idx = len(points) // 2

        # Create a node for the median point
        node = KDTreeNode(
            point=points.iloc[median_idx].values.tolist(),
            axis=axis
        )

        # Recursively build left and right subtrees
        node.left = self._build(points.iloc[:median_idx], depth + 1)
        node.right = self._build(points.iloc[median_idx + 1:], depth + 1)

        return node

    def build(self):
        self.root = self._build(self.data, depth=0)

    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def _knn(self, node, target, k, heap):
        if node is None:
            return

        # Calculate distance from the target point to the current node
        dist = self._distance(target, node.point)

        # Add the current point to the heap (negative distance for max-heap behavior)
        heapq.heappush(heap, (-dist, node.point))
        if len(heap) > k:
            heapq.heappop(heap)  # Remove the farthest point if we exceed k neighbors

        # Determine which subtree to explore first
        axis = node.axis
        next_branch = node.left if target[axis] < node.point[axis] else node.right
        opposite_branch = node.right if target[axis] < node.point[axis] else node.left

        # Recursively search the next branch
        self._knn(next_branch, target, k, heap)

        # Check if the opposite branch could contain closer points
        if abs(target[axis] - node.point[axis]) < -heap[0][0]:
            self._knn(opposite_branch, target, k, heap)

    def knn(self, target, k):
        """Find the k-nearest neighbors to the target point."""
        heap = []  # Max-heap to store the k closest points
        self._knn(self.root, target, k, heap)
        return [point for _, point in sorted(heap, reverse=True)]


def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def main():
    file_path = r"/Users/md.rakibulislam/IIT Life/All Semester in IIT Life/5th Semester/5th Semester Lab Codes/CSE-504/DBMS-2_Lab/KD Tree/IRIS.csv"

    
    try:
        # Load the dataset
        iris_df = load_data(file_path)
        
        # Extract features and drop the target column (species)
        features = iris_df.drop(columns='species')
        
        # print(features)

        # Target point and k
        target_point = list(map(float, input("Enter the target point as space-separated values (e.g., 5.5 3.5 1.3 0.2): ").split()))
        k = int(input("Enter the number of nearest neighbors (k): "))

        # Create KD-Tree
        kd = KD_Tree(features)
        kd.build()

        # Find the k nearest neighbors to the target point
        neighbors = kd.knn(target_point, k)
        print("k-Nearest Neighbors:")
        for neighbor in neighbors:
            print(neighbor)
    
    except FileNotFoundError:
        print("The file path is incorrect. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()