import pandas as pd
import heapq
from math import sqrt
from typing import List, Optional, Tuple

class KDTreeNode:
    """A node in the KD-tree."""
    def __init__(self, point: List[float], axis: int):
        self.point = point  # Point coordinates
        self.axis = axis    # Split axis (0=SepalLength, 1=SepalWidth, etc.)
        self.left: Optional[KDTreeNode] = None   
        self.right: Optional[KDTreeNode] = None  

class KD_Tree:
    """KD-tree implementation with k-nearest neighbor search capability."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize KD-tree with input data.
        
        Args:
            data: DataFrame containing point coordinates
        """
        self.data = data
        self.root: Optional[KDTreeNode] = None
        self.dimensions = len(data.columns)

    def _build(self, points: pd.DataFrame, depth: int) -> Optional[KDTreeNode]:
        """
        Recursively build the KD-tree.
        
        Args:
            points: DataFrame of points to build the tree from
            depth: Current depth in the tree
            
        Returns:
            KDTreeNode or None if points is empty
        """
        if len(points) == 0:
            return None

        axis = depth % self.dimensions
        column = points.columns[axis]

        # Sort and find median
        points = points.sort_values(by=[column])
        median_idx = len(points) // 2

        # Create node and build subtrees
        node = KDTreeNode(
            point=points.iloc[median_idx].values.tolist(),
            axis=axis
        )
        
        node.left = self._build(points.iloc[:median_idx], depth + 1)
        node.right = self._build(points.iloc[median_idx + 1:], depth + 1)

        return node

    def build(self):
        """Build the KD-tree from the input data."""
        self.root = self._build(self.data, depth=0)

    @staticmethod
    def _distance(point1: List[float], point2: List[float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point coordinates
            point2: Second point coordinates
            
        Returns:
            Euclidean distance between the points
        """
        return sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def _knn(self, node: Optional[KDTreeNode], target: List[float], k: int, 
             heap: List[Tuple[float, List[float]]]):
        """
        Recursively find k-nearest neighbors.
        
        Args:
            node: Current node in traversal
            target: Target point to find neighbors for
            k: Number of neighbors to find
            heap: Max-heap of current k-nearest neighbors
        """
        if node is None:
            return

        # Calculate distance and update heap
        dist = self._distance(target, node.point)
        heapq.heappush(heap, (-dist, node.point))
        if len(heap) > k:
            heapq.heappop(heap)

        # Determine search order
        axis = node.axis
        next_branch = node.left if target[axis] < node.point[axis] else node.right
        opposite_branch = node.right if target[axis] < node.point[axis] else node.left

        # Search nearest subtree
        self._knn(next_branch, target, k, heap)

        # Check if we need to search the opposite subtree
        if len(heap) < k or abs(target[axis] - node.point[axis]) < -heap[0][0]:
            self._knn(opposite_branch, target, k, heap)

    def knn(self, target: List[float], k: int) -> List[List[float]]:
        """
        Find k-nearest neighbors to target point.
        
        Args:
            target: Target point coordinates
            k: Number of neighbors to find
            
        Returns:
            List of k-nearest neighbor points, sorted by distance
        """
        heap = []
        self._knn(self.root, target, k, heap)
        return [point for _, point in sorted(heap, reverse=True)]

def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    try:
        df = pd.read_csv(file_path)
        features = df.drop(columns='species')
        labels = df['species']
        return features, labels
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def main():
    """Main function to demonstrate KD-tree usage."""
    try:
        # Load and prepare data
        features, labels = load_and_process_data("IRIS.csv")
        
        # print(labels)
        
        # Get user input
        print("\nEnter target point coordinates (4 values for Iris dataset):")
        target_point = [float(x) for x in input("Format - sepal_length sepal_width petal_length petal_width: ").split()]
        k = int(input("Enter number of nearest neighbors (k): "))
        
        # Validate inputs
        if len(target_point) != 4:
            raise ValueError("Target point must have 4 coordinates for Iris dataset")
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Build and query KD-tree
        kd_tree = KD_Tree(features)
        kd_tree.build()
        neighbors = kd_tree.knn(target_point, k)
        
        # Display results
        print("\nNearest neighbors (in order of increasing distance):")
        print("Format: [sepal_length, sepal_width, petal_length, petal_width]")
        for i, neighbor in enumerate(neighbors, 1):
            distance = KD_Tree._distance(target_point, neighbor)
            print(f"{i}. Distance: {distance:.2f}, Point: {neighbor}")
            
    except ValueError as e:
        print(f"Input error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()