import numpy as np
import pandas as pd
import heapq

class KDTreeNode:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self):
        self.root = None

    def _build_tree(self, points, labels, depth=0):
        if len(points) == 0:
            return None

        k = points.shape[1]
        axis = depth % k

        sorted_idx = points[:, axis].argsort()
        points = points[sorted_idx]
        labels = labels[sorted_idx]

        median_idx = len(points) // 2

        node = KDTreeNode(
            point=points[median_idx],
            label=labels[median_idx],
            axis=axis
        )

        node.left = self._build_tree(points[:median_idx], labels[:median_idx], depth + 1)
        node.right = self._build_tree(points[median_idx + 1:], labels[median_idx + 1:], depth + 1)

        return node

    def build(self, points, labels):
        self.root = self._build_tree(points, labels)

    def _find_k_nearest_neighbors(self, node, target_point, k, depth=0, heap=None):
        if node is None:
            return

        if heap is None:
            heap = []

        current_distance = np.sqrt(np.sum((node.point - target_point) ** 2))

        # Use id(node) to avoid comparing KDTreeNode instances
        if len(heap) < k:
            heapq.heappush(heap, (-current_distance, id(node), node))
        elif current_distance < -heap[0][0]:
            heapq.heappushpop(heap, (-current_distance, id(node), node))

        axis = node.axis
        next_node = node.left if target_point[axis] < node.point[axis] else node.right
        other_node = node.right if target_point[axis] < node.point[axis] else node.left

        self._find_k_nearest_neighbors(next_node, target_point, k, depth + 1, heap)

        if abs(target_point[axis] - node.point[axis]) < -heap[0][0] if heap else 0:
            self._find_k_nearest_neighbors(other_node, target_point, k, depth + 1, heap)

        return heap

    def find_k_nearest_neighbors(self, target_point, k):
        heap = self._find_k_nearest_neighbors(self.root, target_point, k)
        # Extract nodes from the heap entries
        neighbors = [(node.point, node.label, -distance) for distance, _, node in sorted(heap, reverse=True)]
        return neighbors

def load_data_and_build_tree(dataset_path, label_column):
    df = pd.read_csv(dataset_path)
    feature_columns = [col for col in df.columns if col != label_column]
    features = df[feature_columns].values
    labels = df[label_column].values

    kd_tree = KDTree()
    kd_tree.build(features, labels)
    return kd_tree

def main():
    dataset_path = 'IRIS.csv'
    label_column = 'species'
    kd_tree = load_data_and_build_tree(dataset_path, label_column)

    while True:
        try:
            print("\nEnter the test point as values (e.g., 5.0 3.0 4.0 1.0):")
            user_input = input()
            target_point = np.array(list(map(float, user_input.split())))

            print("Enter the number of nearest neighbors you want to find (k):")
            k = int(input())

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

if __name__ == "__main__":
    main()