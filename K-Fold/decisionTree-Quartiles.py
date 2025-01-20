import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, f1_score, fbeta_score

def split_data_quartile(data, split_column):
    q1, q2, q3 = np.percentile(data[:, split_column], [25, 50, 75])
    data_q1 = data[data[:, split_column] <= q1]
    data_q2 = data[(data[:, split_column] > q1) & (data[:, split_column] <= q2)]
    data_q3 = data[(data[:, split_column] > q2) & (data[:, split_column] <= q3)]
    data_q4 = data[data[:, split_column] > q3]
    return data_q1, data_q2, data_q3, data_q4, q1, q2, q3

def determine_best_split_quartile(data):
    best_gini = float('inf')
    best_split = None
    
    for column in range(data.shape[1] - 1):
        q1, q2, q3 = np.percentile(data[:, column], [25, 50, 75])
        splits = [
            (data[data[:, column] <= q1], data[data[:, column] > q1], q1),
            (data[data[:, column] <= q2], data[data[:, column] > q2], q2),
            (data[data[:, column] <= q3], data[data[:, column] > q3], q3)
        ]
        
        for below, above, value in splits:
            if len(below) == 0 or len(above) == 0:
                continue
            
            gini = (len(below) * calculate_gini(below) + 
                   len(above) * calculate_gini(above)) / len(data)
            
            if gini < best_gini:
                best_gini = gini
                best_split = (column, value)
    
    return best_split

def calculate_gini(data):
    _, counts = np.unique(data[:, -1], return_counts=True)
    probabilities = counts / len(data)
    return 1 - np.sum(probabilities ** 2)

def decision_tree_quartile(df, counter=0, min_samples=5, max_depth=3):
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df
        
    if len(np.unique(data[:, -1])) == 1 or len(data) < min_samples or counter == max_depth:
        unique_classes, counts = np.unique(data[:, -1], return_counts=True)
        return species_reverse_mapping[unique_classes[counts.argmax()]]
    
    split_column, split_value = determine_best_split_quartile(data)
    if split_column is None:
        unique_classes, counts = np.unique(data[:, -1], return_counts=True)
        return species_reverse_mapping[unique_classes[counts.argmax()]]
    
    counter += 1
    question = f"{COLUMN_HEADERS[split_column]} <= {split_value:.2f}"
    
    data_below = data[data[:, split_column] <= split_value]
    data_above = data[data[:, split_column] > split_value]
    
    sub_tree = {question: []}
    sub_tree[question] = [
        decision_tree_quartile(data_below, counter, min_samples, max_depth),
        decision_tree_quartile(data_above, counter, min_samples, max_depth)
    ]
    
    return sub_tree

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"{indent}[True] {tree}")
        return
    
    question = list(tree.keys())[0]
    print(f"{indent}{question}")
    yes_answer, no_answer = tree[question]
    print(indent + "        [True] ", end="")
    print_tree(yes_answer, indent + "                ")
    print(indent + "        [False] ", end="")
    print_tree(no_answer, indent + "                ")

def classify_example(example, tree):
    if not isinstance(tree, dict):
        return tree
    question = list(tree.keys())[0]
    feature_name, _, value = question.split(" ")
    
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    return classify_example(example, answer)

# Load data and setup
iris = pd.read_csv("IRIS.csv")
species_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
species_reverse_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
iris["species"] = iris["species"].map(species_mapping)

# Perform k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
metrics = []

for fold, (train_idx, test_idx) in enumerate(kf.split(iris), 1):
    train_data = iris.iloc[train_idx]
    test_data = iris.iloc[test_idx]
    
    tree = decision_tree_quartile(train_data)
    print(f"\nFold {fold} Decision Tree:")
    print_tree(tree)
    
    y_true = test_data["species"].map(species_reverse_mapping)
    y_pred = test_data.apply(lambda x: classify_example(x, tree), axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    
    print("\nConfusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Measure: {f1:.2f}")
    print(f"F2 Measure: {f2:.2f}")
    print("-----------------------------")
    
    metrics.append([accuracy, precision, f1, f2])

avg_metrics = np.mean(metrics, axis=0)
print("\nAverage Metrics Across All Folds:")
print(f"Average Accuracy: {avg_metrics[0]:.2f}%")
print(f"Average Precision: {avg_metrics[1]:.2f}")
print(f"Average F1 Measure: {avg_metrics[2]:.2f}")
print(f"Average F2 Measure: {avg_metrics[3]:.2f}")