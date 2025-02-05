import math
import matplotlib.pyplot as plt

# Load and parse the dataset manually
data = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3,1.4,0.1,Iris-setosa
4.3,3,1.1,0.1,Iris-setosa
5.8,4,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5,3,1.6,0.2,Iris-setosa
5,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
4.4,3,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5,3.3,1.4,0.2,Iris-setosa
7,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5,2,3.5,1,Iris-versicolor
5.9,3,4.2,1.5,Iris-versicolor
6,2.2,4,1,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3,5,1.7,Iris-versicolor
6,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6,2.7,5.1,1.6,Iris-versicolor
5.4,3,4.5,1.5,Iris-versicolor
6,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3,4.1,1.3,Iris-versicolor
5.5,2.5,4,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3,4.6,1.4,Iris-versicolor
5.8,2.6,4,1.2,Iris-versicolor
5,2.3,3.3,1,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
6.5,3,5.8,2.2,Iris-virginica
7.6,3,6.6,2.1,Iris-virginica
4.9,2.5,4.5,1.7,Iris-virginica
7.3,2.9,6.3,1.8,Iris-virginica
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3,5.5,2.1,Iris-virginica
5.7,2.5,5,2,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.4,3.2,5.3,2.3,Iris-virginica
6.5,3,5.5,1.8,Iris-virginica
7.7,3.8,6.7,2.2,Iris-virginica
7.7,2.6,6.9,2.3,Iris-virginica
6,2.2,5,1.5,Iris-virginica
6.9,3.2,5.7,2.3,Iris-virginica
5.6,2.8,4.9,2,Iris-virginica
7.7,2.8,6.7,2,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica
7.2,3,5.8,1.6,Iris-virginica
7.4,2.8,6.1,1.9,Iris-virginica
7.9,3.8,6.4,2,Iris-virginica
6.4,2.8,5.6,2.2,Iris-virginica
6.3,2.8,5.1,1.5,Iris-virginica
6.1,2.6,5.6,1.4,Iris-virginica
7.7,3,6.1,2.3,Iris-virginica
6.3,3.4,5.6,2.4,Iris-virginica
6.4,3.1,5.5,1.8,Iris-virginica
6,3,4.8,1.8,Iris-virginica
6.9,3.1,5.4,2.1,Iris-virginica
6.7,3.1,5.6,2.4,Iris-virginica
6.9,3.1,5.1,2.3,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
6.8,3.2,5.9,2.3,Iris-virginica
6.7,3.3,5.7,2.5,Iris-virginica
6.7,3,5.2,2.3,Iris-virginica
6.3,2.5,5,1.9,Iris-virginica
6.5,3,5.2,2,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3,5.1,1.8,Iris-virginica
"""

# Parse the data
lines = [line.split(',') for line in data.strip().split('\n')]
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
        # Transpose to group features
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

# Predict all instances (for demonstration)
predictions = [predict_instance(summaries, priors, instance) for instance in X]

# Calculate accuracy
accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualization: Plot petal length vs petal width with actual species
plt.figure(figsize=(10, 6))
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
petal_length = [x[2] for x in X]
petal_width = [x[3] for x in X]

# Plot each species
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