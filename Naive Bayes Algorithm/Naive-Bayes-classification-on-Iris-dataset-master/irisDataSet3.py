import math
import matplotlib.pyplot as plt

# Load data from CSV file
def load_csv(filename):
    with open(filename, 'r') as f:
        lines = [line.strip().split(',') for line in f]
    return lines

# Parse the dataset
lines = load_csv('IRIS.csv')
header = lines[0]
rows = lines[1:]

X = []  # Features: sepal_length, sepal_width, petal_length, petal_width
y = []  # Species labels

for row in rows:
    features = list(map(float, row[:4]))
    label = row[4]
    X.append(features)
    y.append(label)

# Split data by class
def separate_by_class(X, y):
    separated = {}
    for i in range(len(y)):
        features = X[i]
        label = y[i]
        if label not in separated:
            separated[label] = []
        separated[label].append(features)
    return separated

separated = separate_by_class(X, y)

# Calculate mean and standard deviation for each feature in each class
def summarize_dataset(separated):
    summaries = {}
    for label, instances in separated.items():
        features_zip = list(zip(*instances))
        class_summaries = []
        for feature in features_zip:
            mean = sum(feature) / len(feature)
            variance = sum((x - mean)**2 for x in feature) / (len(feature) - 1)
            std = math.sqrt(variance)
            class_summaries.append((mean, std))
        summaries[label] = class_summaries
    return summaries

summaries = summarize_dataset(separated)

# Calculate class priors (probabilities)
priors = {label: len(instances)/len(X) for label, instances in separated.items()}

# Gaussian probability density function
def calculate_probability(x, mean, std):
    if std == 0:
        return 0.0  # Handle zero variance
    exponent = math.exp(-((x - mean)**2 / (2 * std**2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

# Predict the class for a single instance
def predict_instance(summaries, priors, instance):
    probabilities = {}
    for label, class_summaries in summaries.items():
        probabilities[label] = priors[label]
        for i in range(len(class_summaries)):
            mean, std = class_summaries[i]
            x = instance[i]
            probabilities[label] *= calculate_probability(x, mean, std)
    return max(probabilities, key=lambda k: probabilities[k])

# Predict all instances
predictions = [predict_instance(summaries, priors, instance) for instance in X]

# Calculate accuracy
accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Confusion Matrix
classes = sorted(list(set(y)))
confusion_matrix = {true_label: {pred_label: 0 for pred_label in classes} for true_label in classes}

for true, pred in zip(y, predictions):
    confusion_matrix[true][pred] += 1

print("Confusion Matrix:")
print("Actual \\ Predicted", end="")
for pred in classes:
    print(f"\t{pred[:15]}", end="")  # Truncate for display
print()
for true in classes:
    print(f"{true[:15]:<15}", end="")
    for pred in classes:
        print(f"\t{confusion_matrix[true][pred]:<5}", end="")
    print()

# Classification Report
print("\nClassification Report:")
print("{:<15} {:<10} {:<10} {:<10} {:<10}".format('Class', 'Precision', 'Recall', 'F1-Score', 'Support'))
for cls in classes:
    TP = confusion_matrix[cls][cls]
    FP = sum(confusion_matrix[other][cls] for other in classes if other != cls)
    FN = sum(confusion_matrix[cls][other] for other in classes if other != cls)
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    support = TP + FN
    
    print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(cls, precision, recall, f1, support))

# Visualization: Plot petal length vs petal width with actual species
# Add this to debug labels
print("Unique labels in y:", set(y))  # Check the actual labels

# Update colors based on the actual labels
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}  # Example adjustment

# Visualization code remains the same
plt.figure(figsize=(10, 6))
petal_length = [x[2] for x in X]
petal_width = [x[3] for x in X]

for label in colors:
    indices = [i for i, lbl in enumerate(y) if lbl == label]
    plt.scatter(
        [petal_length[i] for i in indices],
        [petal_width[i] for i in indices],
        color=colors[label],
        label=label,
        alpha=0.7
    )

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset: Actual Species by Petal Dimensions')
plt.legend()
plt.show()