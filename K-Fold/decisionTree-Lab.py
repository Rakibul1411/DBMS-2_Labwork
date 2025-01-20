import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, f1_score, fbeta_score


# Common Functions
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    return len(unique_classes) == 1


def classify_data(data, species_reverse_mapping):
    label_column = data[:, -1]
    unique_classes, counts = np.unique(label_column, return_counts=True)
    return species_reverse_mapping[unique_classes[counts.argmax()]]


def split_data(data, split_column, split_value):
    data_below = data[data[:, split_column] <= split_value]
    data_above = data[data[:, split_column] > split_value]
    return data_below, data_above


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


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')

    n_classes = len(cm)
    normalized = [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(n_classes)]
    d2h = np.sqrt(np.sum([(1 - x) ** 2 for x in normalized])) / np.sqrt(n_classes)

    return cm, accuracy, precision, f1, f2, d2h


# Gini Method
def gini_method(df, min_samples=2, max_depth=3):
    def calculate_gini(data):
        label_column = data[:, -1]
        _, counts = np.unique(label_column, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def determine_best_split(data):
        best_split = {'column': None, 'value': None, 'gini': float('inf')}

        for column in range(data.shape[1] - 1):
            unique_values = np.unique(data[:, column])
            for value in unique_values:
                data_below, data_above = split_data(data, column, value)
                if len(data_below) == 0 or len(data_above) == 0:
                    continue

                gini = (len(data_below) * calculate_gini(data_below) +
                        len(data_above) * calculate_gini(data_above)) / len(data)

                if gini < best_split['gini']:
                    best_split['column'] = column
                    best_split['value'] = value
                    best_split['gini'] = gini

        return best_split['column'], best_split['value']

    def decision_tree_algorithm(data, counter=0):
        if counter == 0:
            global COLUMN_HEADERS
            COLUMN_HEADERS = df.columns
            data = df.values

        if check_purity(data) or len(data) < min_samples or counter == max_depth:
            return classify_data(data, species_reverse_mapping)

        split_column, split_value = determine_best_split(data)
        if split_column is None:
            return classify_data(data, species_reverse_mapping)

        counter += 1
        data_below, data_above = split_data(data, split_column, split_value)

        question = f"{COLUMN_HEADERS[split_column]} <= {split_value}"
        sub_tree = {question: []}

        yes_answer = decision_tree_algorithm(data_below, counter)
        no_answer = decision_tree_algorithm(data_above, counter)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question] = [yes_answer, no_answer]

        return sub_tree

    return decision_tree_algorithm(df)


# Information Gain Method
def info_gain_method(df, min_samples=2, max_depth=3):
    def calculate_entropy(data):
        label_column = data[:, -1]
        _, counts = np.unique(label_column, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy

    def determine_best_split(data):
        best_split = {'column': None, 'value': None, 'entropy': float('inf')}

        for column in range(data.shape[1] - 1):
            unique_values = np.unique(data[:, column])
            for value in unique_values:
                data_below, data_above = split_data(data, column, value)
                if len(data_below) == 0 or len(data_above) == 0:
                    continue

                overall_entropy = (len(data_below) * calculate_entropy(data_below) +
                                   len(data_above) * calculate_entropy(data_above)) / len(data)

                if overall_entropy < best_split['entropy']:
                    best_split['column'] = column
                    best_split['value'] = value
                    best_split['entropy'] = overall_entropy

        return best_split['column'], best_split['value']

    def decision_tree_algorithm(data, counter=0):
        if counter == 0:
            global COLUMN_HEADERS
            COLUMN_HEADERS = df.columns
            data = df.values

        if check_purity(data) or len(data) < min_samples or counter == max_depth:
            return classify_data(data, species_reverse_mapping)

        split_column, split_value = determine_best_split(data)
        if split_column is None:
            return classify_data(data, species_reverse_mapping)

        counter += 1
        data_below, data_above = split_data(data, split_column, split_value)

        question = f"{COLUMN_HEADERS[split_column]} <= {split_value}"
        sub_tree = {question: []}

        yes_answer = decision_tree_algorithm(data_below, counter)
        no_answer = decision_tree_algorithm(data_above, counter)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question] = [yes_answer, no_answer]

        return sub_tree

    return decision_tree_algorithm(df)


# Main Function
def main():
    # Load and prepare data
    iris = pd.read_csv("IRIS.csv")
    global species_reverse_mapping
    species_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    species_reverse_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    iris["species"] = iris["species"].map(species_mapping)

    # User choice for method
    choice = input("Choose method: (1) Gini or (2) Information Gain: ").strip()
    if choice == "1":
        print("Using Gini method...")
        method = gini_method
    elif choice == "2":
        print("Using Information Gain method...")
        method = info_gain_method
    else:
        print("Invalid choice!")
        return

    # Perform k-fold cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(iris), 1):
        train_data = iris.iloc[train_idx]
        test_data = iris.iloc[test_idx]

        tree = method(train_data)
        print(f"\nFold {fold} Decision Tree:")
        print_tree(tree)

        y_true = test_data["species"].map(species_reverse_mapping)
        y_pred = test_data.apply(lambda x: classify_example(x, tree), axis=1)

        cm, accuracy, precision, f1, f2, d2h = calculate_metrics(y_true, y_pred)
        fold_metrics.append((accuracy, precision, f1, f2, d2h))

        print("\nConfusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"F1 Measure: {f1:.2f}")
        print(f"F2 Measure: {f2:.2f}")
        print(f"Distance to Heaven (d2h): {d2h:.2f}")
        print("-----------------------------")

    # Calculate and print average metrics
    avg_metrics = np.mean(fold_metrics, axis=0)
    print("\nAverage Metrics Across All Folds:")
    print(f"Average Accuracy: {avg_metrics[0]:.2f}%")
    print(f"Average Precision: {avg_metrics[1]:.2f}")
    print(f"Average F1 Measure: {avg_metrics[2]:.2f}")
    print(f"Average F2 Measure: {avg_metrics[3]:.2f}")
    print(f"Average Distance to Heaven: {avg_metrics[4]:.2f}")


if __name__ == "__main__":
    main()
