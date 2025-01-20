import csv
import random
import math

# Constants
MINIMUM_SAMPLE_SIZE = 4
MAX_TREE_DEPTH = 3

class TreeNode:
    def __init__(self, dataset, attribute_list, attribute_values, depth):
        self.dataset = dataset
        self.attribute_list = attribute_list
        self.attribute_values = attribute_values
        self.depth = depth
        self.is_leaf = False
        self.split_attribute = None
        self.split = None
        self.left_child = None
        self.right_child = None
        self.prediction = None

    def build(self):
        if self.depth < MAX_TREE_DEPTH and len(self.dataset) >= MINIMUM_SAMPLE_SIZE and len(set([elem["species"] for elem in self.dataset])) > 1:
            max_gain, attribute, split = self.calculate_max_information_gain()
            if max_gain > 0:
                self.split = split
                self.split_attribute = attribute
                self.split_dataset_and_create_children()
            else:
                self.is_leaf = True
        else:
            self.is_leaf = True

        if self.is_leaf:
            self.set_prediction()

    def calculate_max_information_gain(self):
        max_gain = 0
        best_attribute = None
        best_split = None
        for attribute in self.attribute_list:
            for split in self.attribute_values[attribute]:
                gain = self.calculate_info_gain(attribute, split)
                if gain >= max_gain:
                    max_gain = gain
                    best_attribute = attribute
                    best_split = split
        return max_gain, best_attribute, best_split

    def calculate_info_gain(self, attribute, split):
        set_smaller = [elem for elem in self.dataset if elem[attribute] < split]
        p_smaller = len(set_smaller) / len(self.dataset)
        set_greater_equals = [elem for elem in self.dataset if elem[attribute] >= split]
        p_greater_equals = len(set_greater_equals) / len(self.dataset)
        info_gain = self.calculate_entropy(self.dataset)
        info_gain -= p_smaller * self.calculate_entropy(set_smaller)
        info_gain -= p_greater_equals * self.calculate_entropy(set_greater_equals)
        return info_gain

    def calculate_entropy(self, dataset):
        if len(dataset) == 0:
            return 0
        target_attribute_name = "species"
        target_attribute_values = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        data_entropy = 0
        for val in target_attribute_values:
            p = len([elem for elem in dataset if elem[target_attribute_name] == val]) / len(dataset)
            if p > 0:
                data_entropy += -p * math.log(p, 2)
        return data_entropy

    def split_dataset_and_create_children(self):
        training_set_l = [elem for elem in self.dataset if elem[self.split_attribute] < self.split]
        training_set_r = [elem for elem in self.dataset if elem[self.split_attribute] >= self.split]
        self.left_child = TreeNode(training_set_l, self.attribute_list, self.attribute_values, self.depth + 1)
        self.right_child = TreeNode(training_set_r, self.attribute_list, self.attribute_values, self.depth + 1)
        self.left_child.build()
        self.right_child.build()

    def set_prediction(self):
        species_counts = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
        for elem in self.dataset:
            species_counts[elem["species"]] += 1
        self.prediction = max(species_counts, key=species_counts.get)

    def predict(self, sample):
        if self.is_leaf:
            return self.prediction
        else:
            if sample[self.split_attribute] < self.split:
                return self.left_child.predict(sample)
            else:
                return self.right_child.predict(sample)

    def merge_leaves(self):
        if not self.is_leaf:
            self.left_child.merge_leaves()
            self.right_child.merge_leaves()
            if self.left_child.is_leaf and self.right_child.is_leaf and self.left_child.prediction == self.right_child.prediction:
                self.is_leaf = True
                self.prediction = self.left_child.prediction

    def print(self, prefix):
        if self.is_leaf:
            print("\t" * self.depth + prefix + self.prediction)
        else:
            print("\t" * self.depth + prefix + self.split_attribute + "<" + str(self.split) + "?")
            self.left_child.print("[True] ")
            self.right_child.print("[False] ")

class ID3Tree:
    def __init__(self):
        self.root = None

    def build(self, training_set, attribute_list, attribute_values):
        self.root = TreeNode(training_set, attribute_list, attribute_values, 0)
        self.root.build()

    def merge_leaves(self):
        self.root.merge_leaves()

    def predict(self, sample):
        return self.root.predict(sample)

    def print(self):
        print("----------------")
        print("DECISION TREE")
        self.root.print("")
        print("----------------")

def read_iris_dataset(file_path):
    dataset = []
    with open(file_path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            instance = {
                "sepal_length": float(row[0]),
                "sepal_width": float(row[1]),
                "petal_length": float(row[2]),
                "petal_width": float(row[3]),
                "species": row[4]
            }
            dataset.append(instance)
    return dataset

def main():
    file_path = '/Users/md.rakibulislam/IIT Life/All Semester in IIT Life/5th Semester/5th Semester Lab Codes/CSE-504/DBMS-2_Lab/Iris-Dataset/iris.csv'
    dataset = read_iris_dataset(file_path)

    if not dataset:
        print('Dataset is empty!')
        exit(1)

    accuracy_list = []
    confusion_matrix = {"Iris-setosa": {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0},
                        "Iris-versicolor": {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0},
                        "Iris-virginica": {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}}

    for _ in range(10):
        test_set = []
        species_samples = {species: [] for species in set(instance["species"] for instance in dataset)}
        for instance in dataset:
            species_samples[instance["species"]].append(instance)
        for species, samples in species_samples.items():
            test_set.extend(random.sample(samples, 10))
        training_set = [instance for instance in dataset if instance not in test_set]

        attr_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        attr_domains = {attr: list(set(instance[attr] for instance in dataset)) for attr in attr_list}

        dt = ID3Tree()
        dt.build(training_set, attr_list, attr_domains)
        dt.merge_leaves()

        accuracy = 0
        for sample in test_set:
            prediction = dt.predict(sample)
            if sample["species"] == prediction:
                accuracy += (1 / len(test_set))
            confusion_matrix[sample["species"]][prediction] += 1
        accuracy_list.append(accuracy)
        print("Accuracy on test set: {:.2f}%".format(accuracy * 100))

    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    print("Average accuracy on test set: {:.2f}%".format(avg_accuracy * 100))

    print("\nConfusion Matrix:")
    print("True/Predicted\t\tIris-setosa\tIris-versicolor\tIris-virginica")
    for true_label, predictions in confusion_matrix.items():
        print("{:<16}".format(true_label), end="")
        for predicted_label, count in predictions.items():
            print("\t\t{:<6}".format(count), end="")
        print()

if __name__ == '__main__':
    main()