#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

class CustomKMeans:
    """
    Custom implementation of K-Means clustering algorithm with various initialization methods,
    distance metrics, and mini-batch support.
    """
    def __init__(self, num_clusters=3, max_iter=100, convergence_threshold=1e-4, 
                 num_runs=10, initialization='random', distance_metric='euclidean', 
                 random_state=None, batch_size=100):
        """
        Initialize the CustomKMeans instance.
        
        Parameters:
        -----------
        num_clusters : int
            Number of clusters to form
        max_iter : int
            Maximum number of iterations for a single run
        convergence_threshold : float
            Threshold to determine convergence (minimum distance between old and new centroids)
        num_runs : int
            Number of times the algorithm will be run with different centroid seeds
        initialization : str
            Method to initialize centroids ('random' or 'kmeans++')
        distance_metric : str
            Distance metric ('euclidean' or 'cosine')
        random_state : int or None
            Seed for random number generator
        batch_size : int
            Number of samples to be used in each iteration (mini-batch)
        """
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.num_runs = num_runs
        self.initialization = initialization
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.batch_size = batch_size
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        """
        Initialize cluster centroids using the specified method.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        if self.initialization == 'random':
            # Choose k random samples as centroids   
            self.centroids = X[np.random.choice(len(X), self.num_clusters, replace=False)]
        elif self.initialization == 'kmeans++':
            # Initialize centroids array
            self.centroids = np.zeros((self.num_clusters, X.shape[1]))
            # Choose first centroid randomly
            self.centroids[0] = X[np.random.choice(len(X)), :]
            # Choose remaining centroids using kmeans++ algorithm
            for i in range(1, self.num_clusters):
                # Calculate distance between each sample and nearest existing centroid
                distances = np.array([np.min(np.sum((X - np.array(self.centroids))**2, axis=1)[:i]) for X in X])
                # Calculate probabilities based on distances
                probs = distances / distances.sum()
                # Choose next centroid with probability proportional to distance
                self.centroids[i] = X[np.random.choice(len(X), p=probs)]
        
    def predict_labels(self, X):
        """
        Assign labels to samples based on nearest centroid.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict labels for
        """
        if self.distance_metric == 'cosine':
            # Calculate cosine similarity
            cos_sim = cosine_similarity(X, self.centroids)
            # Assign each sample to the nearest centroid (highest similarity)
            self.labels = np.argmin(1 - cos_sim, axis=1)
        elif self.distance_metric == 'euclidean':
            # Calculate Euclidean distance between each sample and each centroid
            distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=-1)
            # Assign each sample to the nearest centroid
            self.labels = np.argmin(distances, axis=1)

    def fit(self, X):
        """
        Fit the K-means clustering model to the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        best_labels : array, shape (n_samples,)
            Labels of each point for the best run
        best_centroids : array, shape (n_clusters, n_features)
            Centroids found for the best run
        best_SSE : float
            Sum of squared distances for the best run
        SSE_list : list
            List of SSE values for each run
        """
        SSE_list = []
        best_SSE = np.inf
        best_centroids = None
        best_labels = None
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Run the algorithm multiple times and choose the best result    
        for _ in range(self.num_runs):   
            # Initialize the centroids
            self.initialize_centroids(X)
            
            # Run the algorithm for max_iter iterations
            for _ in range(self.max_iter):
                # Select a mini-batch of data points
                batch_indices = np.random.choice(len(X), self.batch_size, replace=False)
                X_batch = X[batch_indices]
                
                # Assign each sample to the nearest centroid
                self.predict_labels(X_batch)
                
                # Update centroids
                new_centroids = np.zeros_like(self.centroids)
                for i in range(self.num_clusters):
                    # Find samples assigned to current centroid
                    assigned_samples = X_batch[self.labels == i]
                    # Handle case where no sample is assigned to the centroid
                    if len(assigned_samples) > 0:
                        new_centroids[i] = assigned_samples.mean(axis=0)
                
                # Calculate distance between old and new centroids
                centroid_dist = np.sqrt(((self.centroids - new_centroids)**2).sum(axis=1))
                self.centroids = new_centroids
                
                # Check for convergence
                if np.all(centroid_dist <= self.convergence_threshold):
                    break
                    
            # Final assignment using all data
            self.predict_labels(X)
            
            # Compute SSE using all data points and final centroids
            SSE = np.sum((X - self.centroids[self.labels])**2)
            SSE_list.append(SSE)
            
            # Keep track of best result
            if SSE < best_SSE:
                best_SSE = SSE
                best_centroids = self.centroids.copy()
                best_labels = self.labels.copy()
                
        return best_labels, best_centroids, best_SSE, SSE_list

def main():
    """
    Main function to execute the image clustering process.
    """
    # Load the image data
    print("Loading data...")
    digit_data = pd.read_csv("test-data-images.txt", header=None)
    
    # Remove columns that only contain zeros
    print("Preprocessing data...")
    non_zero_columns = digit_data.loc[:, ~np.all(digit_data == 0, axis=0)]
    digit_data = non_zero_columns.reset_index(drop=True)
    
    # Dimensionality reduction pipeline
    print("Performing dimensionality reduction...")
    # Step 1: SVD to reduce to 65 components
    svd = TruncatedSVD(n_components=65, random_state=42)
    svd_data = svd.fit_transform(digit_data)
    
    # Step 2: PCA to capture 90% of variance
    pca = PCA(n_components=0.90, random_state=42)
    pca_data = pca.fit_transform(svd_data)
    
    # Step 3: t-SNE for final 2D representation
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2, 
        perplexity=50, 
        n_iter=5000, 
        n_jobs=10, 
        metric='euclidean', 
        random_state=42
    )
    tsne_data = tsne.fit_transform(pca_data)
    
    # Standardize the t-SNE output
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(tsne_data)
    
    # Run K-means for different numbers of clusters
    print("Running K-means clustering...")
    cluster_sse = []
    cluster_labels = []
    cluster_centroids = []
    n_clusters_range = range(2, 21, 2)  # Try 2, 4, 6, ..., 20 clusters
    
    for n_clusters in n_clusters_range:
        print(f"Trying {n_clusters} clusters...")
        kmeans = CustomKMeans(
            num_clusters=n_clusters, 
            max_iter=1000, 
            num_runs=500, 
            initialization='random', 
            distance_metric='euclidean', 
            random_state=1, 
            batch_size=1500
        )
        # Fit the data and get the labels, centroids, and SSE
        labels, centroids, best_SSE, SSE_list = kmeans.fit(data_rescaled)
        cluster_sse.append(best_SSE)
        cluster_labels.append(labels)
        cluster_centroids.append(centroids)
    
    # Find index for K=20 clusters (should be the last element)
    final_cluster_index = len(n_clusters_range) - 1
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, cluster_sse, '-o')
    plt.xticks(n_clusters_range)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('SSE vs Number of Clusters')
    plt.savefig('elbow_curve.png')
    plt.close()
    
    # Plot the final clustering result
    print("Plotting final clustering result...")
    digit_final_labels = [label + 1 for label in cluster_labels[final_cluster_index]]
    
    plt.figure(figsize=(15, 10))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=cluster_labels[final_cluster_index], s=100, cmap='tab10')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f"t-SNE with SSE={cluster_sse[final_cluster_index]}")
    plt.savefig('tsne_clustering.png')
    plt.close()
    
    # Save the clustering results
    output_file = "k_mean_part_2_final.txt"
    with open(output_file, "w") as file:
        for prediction in digit_final_labels:
            file.write(str(prediction) + "\n")
    
    print(f"Clustering results have been saved to {output_file}")

if __name__ == "__main__":
    main()