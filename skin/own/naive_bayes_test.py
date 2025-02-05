import cv2
import numpy as np
import joblib

def extract_features(image):
    """Extracts HSV features (Hue, Saturation, Value) from the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

def detect_skin_with_naive_bayes(image, clf):
    """Detect skin in the image using the trained Naive Bayes classifier."""
    hsv = extract_features(image)
    h, w, _ = hsv.shape
    
    skin_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Predict each pixel using the classifier
    for i in range(h):
        for j in range(w):
            pixel = hsv[i, j].reshape(1, -1)  # Reshape to match input shape for classifier
            prediction = clf.predict(pixel)
            if prediction == 1:
                skin_mask[i, j] = 255  # Mark as skin
    
    # Create an output image where non-skin areas are white and skin areas are kept as is
    output_image = image.copy()
    output_image[skin_mask == 0] = [255, 255, 255]  # Change non-skin areas to white
    
    return output_image, skin_mask

if __name__ == '__main__':
    # Load the trained Naive Bayes model
    clf = joblib.load('naive_bayes_skin_model.pkl')

    # Ask the user for the input image path
    # input_image_path = input("Please enter the path to the image: ")

    # Read the input image
    image = cv2.imread('image7.jpeg')

    if image is not None:
        # Detect skin using the trained Naive Bayes model
        skin_image, mask = detect_skin_with_naive_bayes(image, clf)

        # Display the original image, skin-detected image, and mask
        cv2.imshow("Original Image", image)
        cv2.imshow("Skin Detected Image (Non-skin as white)", skin_image)
        cv2.imshow("Skin Mask", mask)

        # Wait until a key is pressed to close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the results
        cv2.imwrite('detected_skin_image.jpg', skin_image)
        cv2.imwrite('skin_mask.jpg', mask)
    else:
        print("Error: Image not found. Please check the file path.")
