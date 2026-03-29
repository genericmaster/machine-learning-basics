import numpy as np
import matplotlib.pyplot as plt
np.random.seed(312)
dataset=np.random.random_integers(low=0,high=1356,size=(5000,2))#5000 instances with 2 features

#need a functon that  selects the centroid randomly and within the dataset
def Centroid_selector(dataset,k):
    centroids=np.zeros(shape=(k,2))
    for i in range(k):
       choice= np.random.choice(a=dataset.shape[0],replace=False)
       centroids[i]= dataset[choice] #same as centroid[1:,]

    return centroids

#function that finds the euclidean distance between instances and centroids
# this is basically a column appender😂 m soo cooked
def distance(dataset:np.array,centroids:np.array):
   total =0
   distances = np.zeros(shape=(dataset.shape[0], centroids.shape[0]))
   for i in range(0,centroids.shape[0],1):
      for j in range(dataset.shape[1]):
       total += (dataset[:,j]-centroids[i,j])**2 # dataset[:,j]=5000x1 and centroid[i,j] = i
      distances[:,i]=np.sqrt(total)
      total =0
   return distances

#cluster assignment
def assigncluster(dataset,Centroid_array):
   array = distance(dataset,Centroid_array)
   clusters = array.argmin(axis=1)
   return clusters


def computeinertia(dataset,clusters,centroids):
     total =0
     for i in range(centroids.shape[0]):
        points = dataset[clusters==i]
        diff = points -centroids[i]
        total+=np.sum(diff**2)
     return float(total)


#main algorithm
def kmean(dataset,k,max_iter=50):
   inertia_list=[]
   for i in range(1,k):
    Centroid_array=Centroid_selector(dataset,i)
    clusters = assigncluster(dataset,Centroid_array)
    centroids= np.zeros((i,2))
    for iterate in range (max_iter):
     old_cluster = clusters
     for j in range (Centroid_array.shape[0]):   
      updated_centroid= np.mean(a=dataset[clusters==j],axis=0)
      centroids[j]=updated_centroid
     clusters=assigncluster(dataset,centroids)
     if np.array_equal(old_cluster,clusters):
      break  
    inertia = computeinertia(dataset, clusters, centroids)
    inertia_list.append(inertia)
   print(inertia_list)
   return inertia_list

  

