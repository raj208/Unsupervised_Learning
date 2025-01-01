# PCA and K-Means Clustering

This project demonstrates the application of PCA for dimensionality reduction and K-Means clustering for unsupervised learning.

## Steps Implemented:

1. **Data Generation**:
   - Synthetic dataset created using `make_blobs` with 5 features and 4 centers.

2. **Dimensionality Reduction**:
   - Applied PCA to reduce the dataset from 5D to 2D for visualization.

3. **Clustering with K-Means**:
   - Elbow Method used to determine the optimal number of clusters (WCSS).
   - K-Means clustering performed on the training dataset.

4. **Cluster Visualization**:
   - Visualized clusters in 2D space using PCA-transformed data.

5. **Cluster Performance Evaluation**:
   - Used Silhouette Score to evaluate clustering quality.

## Outputs:
- PCA projections of 5D data into 2D.
- Elbow plot to select optimal `k` for K-Means.
- Silhouette coefficients for clustering quality assessment.

![Elbow Plot for Optimal Cluster Selection](screenshot/download2.png)
![Cluster Visualization After PCA (Test Data)](screenshot/download.png)
![Silhouette Coefficients for Cluster Quality Evaluation](screenshot/download1.png)


# PCA and Agglomerative Clustering

This script demonstrates the use of PCA for dimensionality reduction and Agglomerative Clustering for hierarchical clustering.


## Instructions
1. Load the Iris dataset using `sklearn.datasets`.
2. Scale the dataset with `StandardScaler`.
3. Perform PCA to reduce dimensions to 2.
4. Visualize the data in 2D space.
5. Construct a dendrogram for hierarchical clustering.
6. Apply Agglomerative Clustering with Ward linkage.
7. Calculate and visualize silhouette scores to evaluate clustering.

## Understanding PCA
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a new coordinate system where the greatest variance comes first. It helps simplify data visualization and analysis.

### Key Features:
- Reduces high-dimensional data to lower dimensions.
- Retains maximum variance in the dataset.

### Parameters:
- **`n_components`**: Number of components to retain.

### Applications:
- Data visualization.
- Noise reduction.
- Preprocessing for machine learning algorithms.

## Understanding Agglomerative Clustering
Agglomerative Clustering is a hierarchical clustering method that builds clusters by merging smaller clusters iteratively based on a linkage criterion.

### Key Features:
- Builds a tree-like structure (dendrogram).
- Flexible with various linkage methods (e.g., Ward, complete, average).

### Parameters:
- **`n_clusters`**: Number of clusters to form.
- **`linkage`**: Linkage criterion for merging clusters.

### Evaluation:
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to others.

### Applications:
- Gene expression analysis.
- Document classification.
- Customer segmentation.

## Visualizations
- **Dendrogram**: Shows the hierarchical structure of clustering.
- **2D Scatter Plot**: Visualizes clusters after dimensionality reduction.
- **Silhouette Scores Plot**: Evaluates clustering quality for different numbers of clusters.


# Clustering using DBSCAN

This script demonstrates the use of the DBSCAN clustering algorithm on a non-linear dataset.

## Instructions
1. Generate a non-linear dataset using `make_moons`.
2. Scale the dataset with `StandardScaler`.
3. Apply DBSCAN for clustering.
4. Visualize the results.

## Understanding DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points closely packed in space, marking points in low-density areas as outliers. Unlike k-means, DBSCAN does not require the number of clusters to be specified beforehand. 

### Key Features:
- **Density-Based**: Groups points with a minimum number of neighbors within a specified radius (`eps`).
- **Outlier Detection**: Points that don’t belong to any cluster are treated as noise.
- **Non-linear Clustering**: Handles datasets with non-linear patterns effectively.

### Parameters:
- **`eps`**: Maximum distance between points to consider them as neighbors.
- **`min_samples`**: Minimum number of points to form a dense region.

### Advantages:
- No need to predefine the number of clusters.
- Robust to outliers.
- Works well for arbitrarily shaped clusters.

### Disadvantages:
- Sensitive to parameter settings (`eps` and `min_samples`).
- Struggles with clusters of varying densities.



## Requirements
- Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

Install them using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```