#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# In[4]:


X,y = make_blobs(n_samples= 1000, centers = 4, n_features=5, random_state = 102)


# In[5]:


X.shape


# In[13]:


pca = PCA(n_components= 2)
X_pca = pca.fit_transform(X)


# In[15]:


# Plot the first two PCA components
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis')
plt.title('PCA projection of 5D data into 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, _train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[17]:


from sklearn.cluster import KMeans


# In[19]:


#Mannual process
#Elbow methos to select K value


# In[24]:


wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters = k, init='k-means++')
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)


# In[25]:


wcss


# In[36]:


plt.plot(range(1,11), wcss)
plt.xticks(range(1,11))
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# In[42]:


Kmeans = KMeans(n_clusters=3, init="k-means++")


# In[45]:


y_labels = Kmeans.fit_predict(X_train)


# In[49]:


X_pca = pca.fit_transform(X_train)

plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis',c=y_labels)
plt.title('PCA projection of 5D data into 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# In[55]:


y_test_labels = kmeans.predict(X_test)


# In[57]:


X_pca = pca.fit_transform(X_test)

plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis',c=y_test_labels)
plt.title('PCA projection of 5D data into 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# In[58]:


#Automatic process to find cluster


# In[62]:


from kneed import KneeLocator


# In[65]:


kl = KneeLocator(range(1,11), wcss, curve='convex',direction='decreasing')
kl.elbow


# In[66]:


#performance metrics
#silhoutte score
from sklearn.metrics import silhouette_score


# In[72]:


silhouette_coefficients=[]
for k in range (2,11):
    kmeans=KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X_train)
    score=silhouette_score(X_train, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[68]:


silhouette_coefficients


# In[69]:


plt.plot(range(2,11), silhouette_coefficients)
plt.xticks(range(2,11))
plt.xlabel("Number of clusters")
plt.ylabel("Silhoutte coefficient")
plt.show()


# In[ ]:




