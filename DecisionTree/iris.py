import numpy as np
import pandas as pd

class MedianDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """Fit the decision tree using the median as the threshold."""
        y = self._ensure_numeric_labels(y)  # Ensure labels are numeric
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """Build the tree recursively using the median for splitting."""
        if len(set(y)) == 1:
            return {'class': y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': np.bincount(y).argmax()}

        # Randomly select a feature
        feature_idx = np.random.randint(0, X.shape[1])

        # Use the median of the feature for splitting
        median_value = np.median(X[:, feature_idx])

        left_mask = X[:, feature_idx] <= median_value
        right_mask = ~left_mask

        left_y, right_y = y[left_mask], y[right_mask]

        # Handle empty splits by checking length
        if len(left_y) == 0 or len(right_y) == 0:
            return {'class': np.bincount(y).argmax()}

        # Recursively build the tree
        left_tree = self._build_tree(X[left_mask], left_y, depth + 1)
        right_tree = self._build_tree(X[right_mask], right_y, depth + 1)

        return {
            'feature_idx': feature_idx,
            'threshold': median_value,
            'left': left_tree,
            'right': right_tree
        }

    def _ensure_numeric_labels(self, y):
        """Ensure that the labels are numeric, converting them to integers if necessary."""
        if not np.issubdtype(y.dtype, np.integer):
            _, y = np.unique(y, return_inverse=True)  # Convert labels to integer codes
        return y

    def predict(self, X):
        """Predicts the class labels for the input data."""
        return [self._predict_row(x, self.tree) for x in X]

    def _predict_row(self, x, tree):
        """Recursively traverses the tree to make a prediction for a single row."""
        if 'class' in tree:
            return tree['class']
        
        feature_value = x[tree['feature_idx']]
        if feature_value <= tree['threshold']:
            return self._predict_row(x, tree['left'])
        else:
            return self._predict_row(x, tree['right'])

    def accuracy(self, X, y):
        """Calculates the accuracy of the model."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Load the Iris dataset
file_path = 'IRIS.csv'
iris_data = pd.read_csv(file_path)

# Convert the categorical 'species' column to numeric labels
iris_data['species'] = iris_data['species'].astype('category').cat.codes

# Split into features (X) and target (y)
X = iris_data.drop('species', axis=1).values
y = iris_data['species'].values

# Create and train the decision tree with a limited depth
tree = MedianDecisionTree(max_depth=3)
tree.fit(X, y)

# Calculate accuracy on the dataset
accuracy = tree.accuracy(X, y)
print("Accuracy:", accuracy)
