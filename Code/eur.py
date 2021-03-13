# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xarray
from sklearn.cluster import KMeans
import seaborn as sns
import classifiers
from sklearn.model_selection import train_test_split
import metrics
dataset = pd.read_csv('../Dataset/insurance.csv')


# dataset = dataset.fillna(0)
# print(dataset)
dataset.replace({"female": 1, "male": 0, "yes": 1, "no": 0}, inplace=True)
print(dataset)
x = dataset.iloc[:, [0, 1, 2, 3, 4, 6]].values
# x = dataset.iloc[:, [0, 2, 3, 6]].values

# Finding the optimum number of clusters for k-means classification
wcss = []

for i in range(1, 6):
    kmeansOut = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeansOut.fit(x)
    wcss.append(kmeansOut.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 6), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()

# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

print("label del y_means", kmeans.labels_)
dataset["cluster"] = kmeans.labels_
print(dataset)
dataset["cluster"] = dataset["cluster"].map({0: 'Gruppo_2', 1: 'Gruppo_1', 2: 'Gruppo_3'})
print(dataset)
print(kmeans)
#sns.scatterplot(data=dataset, x='age', y='bmi', hue='charges')

#sns.scatterplot(data=dataset, x='age', y='charges', hue='cluster')
sns.scatterplot(data=dataset, x='age', y='charges', hue='bmi')
#sns.scatterplot(data=dataset, x='smoker', y='charges', hue='cluster')

print(dataset.values)
x_data=dataset[dataset.columns[0:5]].to_numpy()
y_data=np.array(dataset["cluster"])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle='true', stratify=y_data)

#naive classificator

classifier1 = classifiers.BayesianClassifier(x_train, y_train)
prediction = classifier1.predict(x_test)
print(classifier1)
metrics.validation(y_test, prediction)

metrics.confusionMatrix(y_test, prediction, name="confusion metrix KNN")



#KNN classificator






# Visualising the clusters
plt.show()
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')

plt.legend()


