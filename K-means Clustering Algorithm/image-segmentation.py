import numpy as np
import cv2
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KMeans:
    def __init__(self, k: int, max_iters: int = 100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.history = []
        self.inertia_ = None
        
    def fit(self, X: np.ndarray) -> np.ndarray:
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            self.history.append(self.centroids.copy())
            
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.k)])
            
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
        
        # Calculate final inertia
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        self.inertia_ = np.sum(np.min(distances, axis=0)**2)
        return labels

def compute_inertias(image_path: str, max_k: int = 10, sample_size: int = 50) -> List[float]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    
    
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        print('HI',indices)
        sampled_pixels = pixels[indices]
    else:
        sampled_pixels = pixels
    
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(k=k)
        kmeans.fit(sampled_pixels)
        inertias.append(kmeans.inertia_)
    return inertias

def plot_elbow(inertias: List[float], max_k: int):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

def detect_skin(image_path: str, k: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    
    kmeans = KMeans(k=k)
    labels = kmeans.fit(pixels)
    
    segmented = labels.reshape(img.shape[:2])
    
    clusters = []
    for i in range(k):
        cluster_mask = (segmented == i).astype(np.uint8)
        cluster = img.copy()
        cluster[cluster_mask == 0] = 0
        clusters.append(cluster)
    
    brightness_scores = []
    for cluster in clusters:
        brightness = np.mean(cv2.cvtColor(cluster, cv2.COLOR_RGB2GRAY))
        brightness_scores.append(brightness)
    
    skin_cluster_idx = np.argmax(brightness_scores)
    skin_mask = (segmented == skin_cluster_idx).astype(np.uint8)
    
    return skin_mask, clusters

def visualize_3d_clustering(img_path: str, k: int, sample_size: int = 1000):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at {img_path} not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    
    indices = np.random.choice(len(pixels), sample_size, replace=False)
    sampled_pixels = pixels[indices]
    
    kmeans = KMeans(k=k)
    labels = kmeans.fit(sampled_pixels)
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2], 
                c=sampled_pixels/255, marker='o', s=20)
    ax1.set_title('Original Points')
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2], c=labels, cmap='viridis', marker='o', s=20, zorder=1)
    ax2.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:, 2], 
                c='red', marker='x', s=400, linewidths=5, zorder=5)
    ax2.set_title('Clustered Points')
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_zlabel('Blue')
    
    plt.tight_layout()
    plt.show()

def visualize_results(img_path: str):
    max_k = 10
    inertias = compute_inertias(img_path, max_k)
    plot_elbow(inertias, max_k)
    k = int(input("Enter the optimal k value based on the elbow plot: "))
    
    skin_mask, clusters = detect_skin(img_path, k)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    skin_detected = img.copy()
    skin_detected[skin_mask == 0] = 0
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(skin_mask, cmap='gray')
    plt.title('Skin Mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(skin_detected)
    plt.title('Detected Skin')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    visualize_3d_clustering(img_path, k)

# Usage
visualize_results('image8.jpeg') 