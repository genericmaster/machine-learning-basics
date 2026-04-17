import numpy as np
import matplotlib.pyplot as plt
from kmean import assigncluster
from greedy_kmean_plus_plus import greedy_kmean_plus_plus
from sklearn.metrics import silhouette_score

CLUSTER_CONFIGS = [
    (-8,  0,  0.00, 6, 0.5, 80),   # horizontal ellipse, left
    ( 8,  0,  0.00, 6, 0.5, 80),   # horizontal ellipse, right
    ( 0, -7,  1.57, 5, 0.5, 80),   # vertical ellipse, bottom
    ( 0,  7,  1.57, 5, 0.5, 80),   # vertical ellipse, top
]

def generate_ellipse_cluster(cx, cy, angle, std_long, std_short, n):
    """Generate points from an elongated (elliptical) cluster."""
    local_x = np.random.normal(0, std_long,  n)
    local_y = np.random.normal(0, std_short, n)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_x = local_x * cos_a - local_y * sin_a
    rot_y = local_x * sin_a + local_y * cos_a
    return np.column_stack([rot_x + cx, rot_y + cy])
 
# --- Generate raw data, ground truth labels discarded
X_parts = []
for (cx, cy, angle, sl, ss, n) in CLUSTER_CONFIGS:
    X_parts.append(generate_ellipse_cluster(cx, cy, angle, sl, ss, n))
 
X = np.vstack(X_parts) 

centroids=greedy_kmean_plus_plus(X,4,10)
clusters=assigncluster(X,centroids)

def intra_cluster(dataset,clusters):
    intra_clusters =[]
    for i in range(dataset.shape[0]):
        data_point =dataset[i]
        cluster_data_point = clusters[i] 
        square = (dataset[(clusters==cluster_data_point )& (np.arange(len(dataset)) != i)]-data_point)**2
        summed = np.sum(square,axis=1)
        distances   = np.sqrt(summed)
        summation=np.sum(distances)
        cluster_mean = summation/(len(dataset[clusters==cluster_data_point])-1)
        intra_clusters.append(float(cluster_mean))
    return intra_clusters

def mean_nearest_cluster_distance(dataset,clusters):
     cluster_distance_list=[]
     minimum_distance_list=[]
     for i in range(dataset.shape[0]):
         data_point =dataset[i]
         new_cluster = clusters[clusters!=clusters[i]]
         for cluster_number in np.unique(new_cluster):
             square=(dataset[clusters==cluster_number]-data_point)**2
             summed = np.sum(square,axis=1)
             distances   = np.sqrt(summed)
             summation=np.sum(distances)
             cluster_mean = summation/len(dataset[clusters==cluster_number])
             cluster_distance_list.append(float(cluster_mean))
         minimum_distance_list.append(min(cluster_distance_list))
         cluster_distance_list=[]
     return minimum_distance_list
        
             

def silhoutee_scorez(a,b):
    summation =0
    for i in range(len(b)):
     score_data_point = (b[i]-a[i])/max(a[i],b[i])
     summation+=score_data_point
    score= summation/len(b)
    return score



def graph(k):
 scores = []
 for i in range(2,12):
    centroids = greedy_kmean_plus_plus(X, i, 10)
    clusters = assigncluster(X, centroids)
    a = intra_cluster(X, clusters)
    b = mean_nearest_cluster_distance(X, clusters)
    score = silhoutee_scorez(a, b)
    scores.append(score)
 return scores
scorez= graph(10)
best_k = np.argmax(scorez) +2
plt.plot(range(2,12),scorez,marker="o")
plt.axvline(x=best_k, color='red', linestyle='--', label=f"Best k={best_k}")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs k")
plt.show()

   
      

    


