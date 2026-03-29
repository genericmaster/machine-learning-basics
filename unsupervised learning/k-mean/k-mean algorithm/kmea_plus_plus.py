
from kmean import assigncluster,dataset,distance,computeinertia,kmean,np


def Kmean_centroid_selector(dataset,k):
    #generate our intial centroid randomlyv and uniformly
    centroids =[]
    distance_matrix=[]
    
    for i in range(0,k):
     if len(centroids)==0:
        index= np.random.choice(a=dataset.shape[0],replace=False)
        centroids.append(dataset[index])
     else:
      centroid_array = np.array(centroids)
      square_difference_matrix = (dataset - centroid_array[i-1])**2
      distance_matrix.append(np.sqrt(np.sum(square_difference_matrix,axis=1)))
      temp=np.array(distance_matrix).T
      D_x= np.min(a=temp,axis=1)
      square_distance = D_x**2
      sum_distances = np.sum(square_distance)

      prob_distribution = square_distance/sum_distances
      indexer = np.random.choice(a=dataset.shape[0],p=prob_distribution,replace=False)
      centroids.append(dataset[indexer])
    
    centroids = np.array(centroids)
    distance_matrix = np.array(distance_matrix)
     
    return centroids
     #return total_distance.shape

def kmean_plus_plus(dataset,k,max_iter=50):
   inertia_list=[]
   for i in range(1,k):
    Centroid_array=Kmean_centroid_selector(dataset,i)
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
   print(f'k_mean++ inertial list:{inertia_list}')
   return inertia_list




