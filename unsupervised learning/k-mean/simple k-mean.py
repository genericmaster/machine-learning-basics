from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt 

X,y = make_blobs(n_samples=500,n_features=5,centers=5,cluster_std=0.8,random_state=0)
#plt.scatter(X[:, 0], X[:, 1], cmap='viridis', s=20)
#plt.title("make_blobs example")
#plt.show()

k=5

k_mean= KMeans(n_clusters=5,random_state=0)
y_pred=k_mean.fit_predict(X)

print(y_pred)

print(y_pred is k_mean.labels_)

print(k_mean.cluster_centers_)

#CREATING A NEW DARASET
x_new = np.array([[12,1,13,7,5],[2,1,3,17,15],[24,3,11,0,4],[67,12,13,7,1],[4,13,1,6,1]])


y1=k_mean.predict(x_new)
scores=k_mean.transform(x_new)
print(scores)