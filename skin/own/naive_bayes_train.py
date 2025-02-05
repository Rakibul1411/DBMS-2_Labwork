import cv2
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(image):
    """Extracts RGB or HSV features (Hue, Saturation, Value) from the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

def generate_training_data(mask_folder, image_folder):
    """Generates feature vectors and labels from the skin mask and real images."""
    skin_features = []
    non_skin_features = []

    # List all files in the mask folder (detected images)
    mask_files = os.listdir(mask_folder)
    print(f"Found {len(mask_files)} mask files in {mask_folder}.")
    
    for mask_file in mask_files:
        # Ensure the corresponding real image exists
        real_image_file = mask_file.replace(".bmp", ".jpg")  # Match mask with real image file

        mask_path = os.path.join(mask_folder, mask_file)
        real_image_path = os.path.join(image_folder, real_image_file)

        # Read the corresponding mask and real image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        real_image = cv2.imread(real_image_path)

        if mask is None or real_image is None:
            print(f"Skipping {mask_file} as one of the images is not loaded correctly.")
            continue

        # Extract features from the real image
        hsv = extract_features(real_image)
        for i in range(hsv.shape[0]):
            for j in range(hsv.shape[1]):
                # If the mask pixel is white (skin), label it as 1 (skin)
                if mask[i, j] == 255:
                    skin_features.append(hsv[i, j])
                else:
                    non_skin_features.append(hsv[i, j])

    # Check if any features were collected
    print(f"Collected {len(skin_features)} skin features and {len(non_skin_features)} non-skin features.")

    # Combine skin and non-skin data
    features = np.array(skin_features + non_skin_features)
    labels = np.array([1] * len(skin_features) + [0] * len(non_skin_features))  # 1 for skin, 0 for non-skin

    return features, labels

def train_naive_bayes_classifier(features, labels):
    """Train Naive Bayes classifier on the given features and labels."""
    clf = GaussianNB()
    clf.fit(features, labels)
    return clf

if __name__ == '__main__':
    # Path to the folders
    mask_folder = 'Mask'  # Folder with detected images (.bmp)
    image_folder = 'ibtd'    # Folder with real images (.jpeg)

    # Generate training data
    features, labels = generate_training_data(mask_folder, image_folder)

    # If no valid data is found, print error and exit
    if features.size == 0 or labels.size == 0:
        print("No valid training data found. Please check your image files and paths.")
        exit()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    # Train Naive Bayes model
    clf = train_naive_bayes_classifier(X_train, y_train)

    # Evaluate the classifier on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the trained model
    joblib.dump(clf, 'naive_bayes_skin_model.pkl')
