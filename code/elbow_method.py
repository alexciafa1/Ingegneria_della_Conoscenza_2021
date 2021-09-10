import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

cluster_dataset = pd.read_csv('../dataset/anime_preprocessato.csv')

# START CLUSTER
# ELBOW METHOD
wcss = []

for i in range(1, 10):

    kmeansOut = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeansOut.fit(cluster_dataset)

    wcss.append(kmeansOut.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 10), wcss, 'bx-')

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.xticks(range(1,10))

plt.ylabel('WCSS')  # within cluster sum of squares

plt.grid(True)

plt.show()

k = 3

# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)

y_kmeans = kmeans.fit_predict(cluster_dataset)

cluster_dataset["cluster"] = kmeans.labels_ + 1



