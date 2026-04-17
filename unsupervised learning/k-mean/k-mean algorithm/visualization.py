import matplotlib.pyplot as plt
from kmean  import dataset,assigncluster,Centroid_selector,kmean
from kmea_plus_plus import kmean_plus_plus
from greedy_kmean_plus_plus import greedy_kmean_plus_plus
from silhoutee_score import X
centroid_array = Centroid_selector(dataset,7)
clusters = assigncluster(dataset,centroid_array)
greedy_kmean_array =greedy_kmean_plus_plus(dataset,7,10)
cluster_2= assigncluster(dataset,greedy_kmean_array)
ellipse_centroid= greedy_kmean_plus_plus(X,4,10)
cluster_3= assigncluster(X,greedy_kmean_array)
plt.figure(figsize=(8,6))
#plt.scatter(x=dataset[:, 0],y=dataset[:, 1],c=clusters)
#a=plt.scatter(x=centroid_array[:, 0],y=centroid_array[:, 1],c='red',marker='x')

#plt.show()
#plt.scatter(x=dataset[:, 0],y=dataset[:, 1],c=cluster_2)
#b=plt.scatter(x=greedy_kmean_array[:,0],y=greedy_kmean_array[:,1], c="red",marker="x")

c= plt.scatter(x=X[:,0],y=X[:,1],c=cluster_3)
plt.scatter(x=ellipse_centroid[:,0],y=ellipse_centroid[:,1],marker="x",c="pink")

c.axes.set_xlabel('feature x')
c.axes.set_ylabel('feature y')



k_values =range(1,20)
inertia_list=kmean(dataset,20)
inertia_list_plus = kmean_plus_plus(dataset,20)
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia_list, marker='o',label='kmean')
plt.show()
plt.plot(k_values, inertia_list_plus, marker='o',label='kmean++')
plt.legend()
plt.xlabel('number of clusters')
plt.ylabel('toot mean square error')
plt.title('cluster perfomance score')

plt.show()
