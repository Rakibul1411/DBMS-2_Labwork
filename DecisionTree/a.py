import numpy as np

class IntuitiveDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """Fit the decision tree using a simplistic approach."""
        y = self._ensure_numeric_labels(y)  # Ensure labels are numeric
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """Build the tree recursively using an unconventional method."""
        if len(set(y)) == 1:
            return {'class': y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': np.bincount(y).argmax()}

        # Randomly select a feature
        feature_idx = np.random.randint(0, X.shape[1])

        # Try a random threshold from the feature's unique values
        possible_thresholds = np.unique(X[:, feature_idx])
        best_split = None
        best_score = float('inf')
        
        # Try multiple thresholds and choose the one with the "simplest" division
        for threshold in possible_thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            left_y, right_y = y[left_mask], y[right_mask]

            # Handle empty splits by checking length
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # Calculate a "simplicity" score based on how similar the target is in the split
            left_score = self._calculate_simplicity_score(left_y)
            right_score = self._calculate_simplicity_score(right_y)
            
            # We want to minimize the "impurity" of the split
            score = left_score + right_score

            if score < best_score:
                best_score = score
                best_split = (threshold, left_mask, right_mask, left_y, right_y)

        if best_split is None:  # If no valid split found, return majority class
            return {'class': np.bincount(y).argmax()}

        # Apply the best split found
        threshold, left_mask, right_mask, left_y, right_y = best_split

        left_tree = self._build_tree(X[left_mask], left_y, depth + 1)
        right_tree = self._build_tree(X[right_mask], right_y, depth + 1)

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _calculate_simplicity_score(self, y):
        """Calculates how 'simple' a group is based on target values (low variance, high homogeneity)."""
        # A simple measure: the number of unique classes in the target. 
        return len(np.unique(y))

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


if __name__ == "__main__":
    # Sample data (features and target labels)
    X = np.array([[2, 3], [4, 5], [7, 8], [6, 1], [3, 4], [1, 2]])
    y = np.array([0, 1, 0, 1, 0, 1])

    # Create and train the decision tree
    tree = IntuitiveDecisionTree(max_depth=3)
    tree.fit(X, y)

    # Calculate accuracy on the training set
    accuracy = tree.accuracy(X, y)
    print("Accuracy:", accuracy)
