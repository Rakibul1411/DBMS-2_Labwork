import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

# Initialize centroids randomly
def initialize_centroids(X, k):
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    return X[random_indices]

# Assign clusters based on the nearest centroid
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = np.linalg.norm(point - centroids, axis=1)
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# Update centroids by calculating the mean of points in each cluster
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(X[np.random.choice(X.shape[0])])  # Handle empty cluster by reinitializing
    return np.array(new_centroids)

# K-Means algorithm
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for iteration in range(max_iters):
        print(f"Iteration {iteration + 1}:")

        clusters = assign_clusters(X, centroids)
        for i in range(k):
            cluster_points = X[clusters == i]
            print(f"Cluster {i + 1}: {cluster_points}")

        new_centroids = update_centroids(X, clusters, k)
        print(f"Centroids: {new_centroids}\n")

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters

# Visualization of clusters and centroids in 3D and 4D
def plot_clusters(X, clusters, centroids):
    fig = plt.figure(figsize=(12, 6))

    # 3D visualization
    ax = fig.add_subplot(121, projection='3d')
    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {i+1} (Size: {len(cluster_points)})", alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=200, label='Centroids')
    ax.set_title("3D K-Means Clustering")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1))

    # 2D Projection with markers
    ax2 = fig.add_subplot(122)
    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1} (Size: {len(cluster_points)})", alpha=0.6)
    ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    ax2.set_title("2D Projection")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load dataset
    file_path = "IRIS.csv"  # Replace with the actual file path if needed
    data = pd.read_csv(file_path)
    
    # Assume the features for clustering are in the first few columns
    X = data.iloc[:, :-1].values
    
    # Allow user to input k or randomly generate k
    user_input = input("Enter the number of clusters (k) or type 'random' to let the program decide: ")
    if user_input.lower() == 'random':
        k = random.randint(2, 6)  # Randomly choose k between 2 and 6
    else:
        k = int(user_input)

    print(f"Number of clusters chosen: {k}")

    # Run k-means clustering
    centroids, clusters = kmeans(X, k)

    # Plot results
    plot_clusters(X, clusters, centroids)

if __name__ == "__main__":
    main()
