# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset[['Annual Income (k$)','Spending Score (1-100)']]

# using dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# visualizing
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidian distance')
plt.show()

# fitting hierarchical clustering to the mall data
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X)

# visualizing the clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_hc)
plt.show()
