import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, f1_score, fbeta_score

# Load the iris dataset from CSV
iris = pd.read_csv("IRIS.csv")

# Encode the species column as numerical values
species_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
iris["species"] = iris["species"].map(species_mapping)

# Function to check purity
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    return False

# Function to classify data
def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    classification = unique_classes[counts_unique_classes.argmax()]
    return classification

# Function to split data based on mean
def split_data_mean(data, split_column):
    mean_value = np.mean(data[:, split_column])
    data_below = data[data[:, split_column] <= mean_value]
    data_above = data[data[:, split_column] > mean_value]
    return data_below, data_above, mean_value

# Function to determine the best split based on mean
def determine_best_split_mean(data):
    _, n_columns = data.shape
    best_split_column = None
    best_split_value = None
    best_variance_reduction = -1

    for column_index in range(n_columns - 1):  # Exclude the label column
        data_below, data_above, mean_value = split_data_mean(data, column_index)
        if len(data_below) == 0 or len(data_above) == 0:
            continue

        # Calculate variance reduction
        overall_variance = np.var(data[:, -1])  # Variance of the target column
        variance_below = np.var(data_below[:, -1])
        variance_above = np.var(data_above[:, -1])
        variance_reduction = overall_variance - (len(data_below) / len(data)) * variance_below - (len(data_above) / len(data)) * variance_above

        if variance_reduction > best_variance_reduction:
            best_variance_reduction = variance_reduction
            best_split_column = column_index
            best_split_value = mean_value

    return best_split_column, best_split_value

# Decision tree algorithm using mean-based splitting
def decision_tree_algorithm_mean(df, counter=0, min_samples=2, max_depth=5):
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df
        
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:
        counter += 1
        split_column, split_value = determine_best_split_mean(data)
        if split_column is None:
            classification = classify_data(data)
            return classification

        data_below, data_above, _ = split_data_mean(data, split_column)
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        yes_answer = decision_tree_algorithm_mean(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm_mean(data_above, counter, min_samples, max_depth)
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        return sub_tree

# Function to classify examples
def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)

# Function to calculate metrics
def calculate_metrics(df, tree):
    df = df.copy()
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    y_true = df["species"]
    y_pred = df["classification"]
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("Confusion Matrix:")
    print(cm)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Precision
    precision = precision_score(y_true, y_pred, average="weighted")
    print(f"Precision: {precision:.2f}")
    
    # F1 Measure
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"F1 Measure: {f1:.2f}")
    
    # F2 Measure
    f2 = fbeta_score(y_true, y_pred, beta=2, average="weighted")
    print(f"F2 Measure: {f2:.2f}")
    
    # Distance to Heaven (1 - Accuracy)
    distance_to_heaven = 1 - accuracy
    print(f"Distance to Heaven: {distance_to_heaven:.2f}")

    return accuracy, precision, f1, f2, distance_to_heaven

# k-Fold Cross-Validation
def k_fold_cross_validation(dataframe, k):
    kf = KFold(n_splits=k, shuffle=True)
    accuracies = []
    precisions = []
    f1_scores = []
    f2_scores = []
    distances_to_heaven = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(dataframe)):
        train_df = dataframe.iloc[train_index]
        test_df = dataframe.iloc[test_index]
        
        tree = decision_tree_algorithm_mean(train_df, max_depth=3)
        print(f"\nFold {fold + 1} Decision Tree:")
        print(tree)
        
        accuracy, precision, f1, f2, distance_to_heaven = calculate_metrics(test_df, tree)
        accuracies.append(accuracy)
        precisions.append(precision)
        f1_scores.append(f1)
        f2_scores.append(f2)
        distances_to_heaven.append(distance_to_heaven)
        print("-----------------------------")
    
    # Calculate averages
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_f1 = np.mean(f1_scores)
    avg_f2 = np.mean(f2_scores)
    avg_distance_to_heaven = np.mean(distances_to_heaven)
    
    print("\nAverage Metrics Across All Folds:")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average F1 Measure: {avg_f1:.2f}")
    print(f"Average F2 Measure: {avg_f2:.2f}")
    print(f"Average Distance to Heaven: {avg_distance_to_heaven:.2f}")

# Main execution
k = 5
k_fold_cross_validation(iris, k)