# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from Code import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Importing dataset

dataset = pd.read_csv('../Dataset/insurance.csv')
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
# sns.scatterplot(data=dataset, x='age', y='bmi', hue='charges')

# sns.scatterplot(data=dataset, x='age', y='charges', hue='cluster')
sns.scatterplot(data=dataset, x='age', y='charges', hue='bmi')
# sns.scatterplot(data=dataset, x='smoker', y='charges', hue='cluster')

print(dataset.values)
x_data = dataset[dataset.columns[0:5]].to_numpy()
y_data = np.array(dataset["cluster"])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle='true', stratify=y_data)


# Function for plotting the results from classification
def plot_results(classifier, X_test, y_test, name):
    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap="Reds");

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title(name);
    plt.show()


# Naive Bayes Classifier

classifier1 = GaussianNB()
classifier1.fit(x_train, y_train)
plot_results(classifier1, x_test, y_test, 'Gaussian Naive Bayes')


# KNN Classifier

classifier2 = KNeighborsClassifier()
# defining parameter range
param_grid = {'n_neighbors': [1, 4, 5, 6, 7, 8],
              'leaf_size': [1, 3, 5, 10],
              }
# Hyperparameters Optimization
grid_knn = GridSearchCV(classifier2, param_grid=param_grid)
grid_knn.fit(x_train, y_train)
print("best_params", grid_knn.best_params_)
print("best_estimator", grid_knn.best_estimator_)
plot_results(grid_knn, x_test, y_test, 'K-nearest neighbors')

# Support Vector Machines
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'sigmoid']}

grid_svm = GridSearchCV(SVC(), param_grid=param_grid, cv=5, refit=True, verbose=False)
grid_svm.fit(x_train, y_train)
print("best_params", grid_svm.best_params_)
print("best_estimator", grid_svm.best_estimator_)
plot_results(grid_svm, x_test, y_test, 'Support Vector Machines')
