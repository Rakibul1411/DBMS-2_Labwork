import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, f1_score, fbeta_score, recall_score

# Load the iris dataset from CSV
iris = pd.read_csv("IRIS.csv")

# Encode the species column as numerical values
species_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
species_reverse_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
iris["species"] = iris["species"].map(species_mapping)

# Function to check purity
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    return len(unique_classes) == 1

# Function to classify data
def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    classification = unique_classes[counts_unique_classes.argmax()]
    return species_reverse_mapping[classification]  # Return species name instead of numerical value

# Function to calculate correlation between features and target
def calculate_correlation(data):
    correlations = []
    n_features = data.shape[1] - 1  # Exclude the target column
    for i in range(n_features):
        correlation = np.corrcoef(data[:, i], data[:, -1])[0, 1]
        correlations.append(abs(correlation))  # Use absolute correlation
    return correlations

# Function to split data based on correlation
def split_data_correlation(data):
    correlations = calculate_correlation(data)
    best_feature = np.argmax(correlations)  # Feature with the highest absolute correlation
    median_value = np.median(data[:, best_feature])
    data_below = data[data[:, best_feature] <= median_value]
    data_above = data[data[:, best_feature] > median_value]
    return data_below, data_above, best_feature, median_value

# Correlation-based decision tree algorithm
def correlation_decision_tree(df, counter=0, min_samples=2, max_depth=5, used_features=None):
    if used_features is None:
        used_features = set()
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    if check_purity(data) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:
        counter += 1
        data_below, data_above, best_feature, median_value = split_data_correlation(data)
        feature_name = COLUMN_HEADERS[best_feature]

        # Prevent redundant splits on the same feature
        if feature_name in used_features:
            classification = classify_data(data)
            return classification
        used_features.add(feature_name)

        question = f"{feature_name} <= {median_value}"
        sub_tree = {question: []}
        yes_answer = correlation_decision_tree(data_below, counter, min_samples, max_depth, used_features)
        no_answer = correlation_decision_tree(data_above, counter, min_samples, max_depth, used_features)
        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)
        return sub_tree

# Function to print the decision tree in the desired format
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + str(tree))
        return
    question = list(tree.keys())[0]
    print(indent + question)
    print(indent + "        [True] ", end="")
    print_tree(tree[question][0], indent + "        ")
    print(indent + "        [False] ", end="")
    print_tree(tree[question][1], indent + "        ")

# Function to classify examples
def classify_example(example, tree):
    if not isinstance(tree, dict):
        return tree
    question = list(tree.keys())[0]
    feature_name, _, value = question.split()
    value = float(value)
    if example[feature_name] <= value:
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    return classify_example(example, answer)

# Function to calculate metrics
def calculate_metrics(df, tree):
    df = df.copy()
    df["classification"] = df.apply(lambda row: classify_example(row, tree), axis=1)
    y_true = df["species"].map(species_reverse_mapping)  # Convert numerical labels to species names
    y_pred = df["classification"]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Precision
    precision = precision_score(y_true, y_pred, average="weighted")

    # Recall (Sensitivity)
    recall = recall_score(y_true, y_pred, average="weighted")

    # F1 Measure
    f1 = f1_score(y_true, y_pred, average="weighted")

    # F2 Measure
    f2 = fbeta_score(y_true, y_pred, beta=2, average="weighted")

    # Distance to Heaven (d2h)
    n_classes = len(cm)  # Number of classes (goals)
    normalized_values = []
    for i in range(n_classes):
        correct_predictions = cm[i, i]
        total_samples = np.sum(cm[i, :])
        normalized_value = correct_predictions / total_samples
        normalized_values.append(normalized_value)

    heaven = [1] * n_classes  # Heaven is [1, 1, 1] for all goals
    squared_differences = [(heaven[i] - normalized_values[i]) ** 2 for i in range(n_classes)]
    euclidean_distance = np.sqrt(np.sum(squared_differences))
    d2h = euclidean_distance / np.sqrt(n_classes)

    return cm, accuracy, precision, recall, f1, f2, d2h

# k-Fold Cross-Validation
def k_fold_cross_validation(dataframe, k):
    kf = KFold(n_splits=k, shuffle=True)
    confusion_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    f2_scores = []
    d2h_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(dataframe)):
        train_df = dataframe.iloc[train_index]
        test_df = dataframe.iloc[test_index]

        # Build the decision tree
        tree = correlation_decision_tree(train_df, max_depth=3)
        print(f"\nFold {fold + 1} Decision Tree:")
        print_tree(tree)

        # Evaluate metrics
        cm, accuracy, precision, recall, f1, f2, d2h = calculate_metrics(test_df, tree)
        confusion_matrices.append(cm)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        f2_scores.append(f2)
        d2h_scores.append(d2h)

        # Print metrics for the current fold
        print(f"Fold {fold + 1} Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Measure: {f1:.2f}")
        print(f"F2 Measure: {f2:.2f}")
        print(f"Distance to Heaven (d2h): {d2h:.2f}")
        print("-----------------------------")

    # Calculate averages
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_f2 = np.mean(f2_scores)
    avg_d2h = np.mean(d2h_scores)

    # Print average performance metrics
    print("\nAverage Performance Metrics Across All Folds:")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1 Measure: {avg_f1:.2f}")
    print(f"Average F2 Measure: {avg_f2:.2f}")
    print(f"Average Distance to Heaven (d2h): {avg_d2h:.2f}")

# Main execution
k = 5
k_fold_cross_validation(iris, k)