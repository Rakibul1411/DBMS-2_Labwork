import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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
            new_centroids.append(X[np.random.choice(X.shape[0])])  
    return np.array(new_centroids)

# K-Means algorithm
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for iteration in range(max_iters):
        print("Iteration: ", iteration)
        clusters = assign_clusters(X, centroids)
        print("Clusters: ", clusters)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Calculate WCSS (Within-Cluster-Sum of Squared Errors)
def calculate_wcss(X, centroids, clusters):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss


def elbow_method(X, max_k=10):
    wcss_values = []
    for k in range(1, max_k + 1):
        centroids, clusters = kmeans(X, k)
        wcss = calculate_wcss(X, centroids, clusters)
        wcss_values.append(wcss)
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), wcss_values, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster-Sum of Squared Errors (WCSS)')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()


def plot_clusters(X, clusters, centroids):
    plt.figure(figsize=(8, 6))
    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1} (Size: {len(cluster_points)})", alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
    plt.tight_layout()
    plt.show()

# Main function
def main():
    file_path = "IRIS.csv"  
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, :-1].values
    
    print("Running elbow method to determine the best number of clusters...")
    elbow_method(X, max_k=10)
    
    k = int(input("Enter the number of clusters (k) based on the elbow plot: "))
    print(f"Number of clusters chosen: {k}")

    centroids, clusters = kmeans(X, k)

    print("Visualizing clusters...")
    plot_clusters(X, clusters, centroids)

if __name__ == "__main__":
    main()