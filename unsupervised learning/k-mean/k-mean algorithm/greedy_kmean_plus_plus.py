from kmea_plus_plus import kmean_plus_plus,np
from kmean import dataset,assigncluster,computeinertia,distance

def greedy_kmean_plus_plus(dataset,k,generator):
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
      indexer = np.random.choice(a=dataset.shape[0],p=prob_distribution,replace=False,size=generator)
      
      centroid_matrix= dataset[indexer]
      #np.set_printoptions(threshold=np.inf)
      inertia_matrix=distance(dataset,centroid_matrix)
      potential_distances=np.minimum(D_x[:, np.newaxis],inertia_matrix)
      inertia = np.sum(potential_distances**2,axis=0)
      minimum = np.argmin(inertia)
      chosen_centroid=dataset[indexer[minimum]]
    
      centroids.append(chosen_centroid)
    
    final_centroids = np.array(centroids)
    distance_matrix = np.array(distance_matrix)
     
    return final_centroids

print(greedy_kmean_plus_plus(dataset,10,10))