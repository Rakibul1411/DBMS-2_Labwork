import numpy as np
import cv2

def kmeans_clustering(data, k, max_iters=100):
    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Compute distances and assign clusters
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)

        # Compute new centroids, handling empty clusters
        new_centroids = []
        for i in range(k):
            cluster_points = data[cluster_assignments == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])  # Keep previous centroid if empty
        
        new_centroids = np.array(new_centroids)
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        
        centroids = new_centroids
    
    return centroids, cluster_assignments

def find_optimal_clusters(data, max_k=10):
    max_k = min(max_k, len(data))
    wcss = []
    
    for k in range(2, max_k + 1):
        _, cluster_assignments = kmeans_clustering(data, k)
        wcss.append(np.sum(np.min(cluster_assignments, axis=0)))  # Within-cluster sum of squares
    
    # Second derivative to find the elbow point
    second_derivative = np.diff(wcss, 2)
    elbow_k = np.argmin(second_derivative) + 2
    
    return elbow_k

def segment_image(image_path, output_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)  # Flatten the image
    
    # Determine optimal clusters
    optimal_k = find_optimal_clusters(pixels, max_k=10)

    # Apply K-Means clustering
    centroids, cluster_assignments = kmeans_clustering(pixels, optimal_k)
    
    # Assign each pixel to its centroid
    segmented_pixels = np.array([centroids[label] for label in cluster_assignments], dtype=np.uint8)
    segmented_image = segmented_pixels.reshape(image_rgb.shape)

    # Save segmented image
    segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, segmented_bgr)

    print(f"Segmented image saved as {output_path}")

if __name__ == "__main__":
    segment_image("image5.jpeg", "segmented_image.jpeg")
