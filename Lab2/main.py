from sklearn.cluster import KMeans
import pandas as pd
import time

start = time.time()
columns = ['1','2','3','4','5']
irisdata = pd.read_csv('iris.data', names=columns)
irisdata['5'] = pd.Categorical(irisdata["5"])
irisdata["5"] = irisdata["5"].cat.codes
X = irisdata.values[:, 0:4]
y = irisdata.values[:, 4]
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
end = time.time() 
print(kmeans.n_iter_)
print(centroids)
print((end - start) * 1000 , " ms")
