Q- To explore unsupervised Machine Learning 
#K-mean clustering

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

#importing dataset
X = iris_df.iloc[:, [0, 1, 2, 3]].values

#using elbow method to find optimal number of clusters 
wscc=[]
from sklearn.cluster import KMeans
for i in range(1,11):
    kmean = KMeans(n_clusters= i , init= 'k-means++' , max_iter= 300 , n_init= 10 , random_state= 0)
    kmean.fit(X)
    wscc.append(kmean.inertia_)
plt.plot(range(1,11),wscc)
plt.title('the elbow method')
plt.xlabel('no.of cluster')
plt.ylabel('wscc')
plt.show()

#Applying k-mean to mall dataset 
kmean= KMeans(n_clusters= 3 , init= 'k-means++' , max_iter= 300 , n_init= 10 , random_state= 0)
y_kmean=kmean.fit_predict(X)

#visualizing the cluster
plt.scatter(X[y_kmean == 0 , 0], X[y_kmean == 0 , 1],s=100 , color = 'red', label= 'Iris-setosa')
plt.scatter(X[y_kmean == 1 , 0], X[y_kmean == 1 , 1],s=100, color = 'green', label= 'Iris-versicolour')
plt.scatter(X[y_kmean == 2 , 0], X[y_kmean == 2 , 1],s=100, color = 'blue', label= 'Iris-virginica')
plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1],s=300, color ='yellow', label='centriod')
plt.title('cluster of clients')
plt.xlabel('annual salary')
plt.ylabel('spending score')
plt.legend()
plt.show()



    
    


