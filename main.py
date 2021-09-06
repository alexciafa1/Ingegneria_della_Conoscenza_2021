import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

anime = pd.read_csv('dataset/anime.csv')

anime.drop('members', axis=1, inplace=True)


def creazionelistGenre(row, list_genre):
    list_ = str(row['genre']).split(', ')
    for genre_ in list_:
        if genre_ not in list_genre:
            list_genre.append(genre_)


list_genre = []
anime.apply(lambda row: creazionelistGenre(row, list_genre), axis=1)
list_genre.remove('nan')

# genre
def creazioneArrayGenre(row, array):
    if row['genre'] is not None:
        array.append(row['genre'])


genre = []

anime.apply(lambda row: creazioneArrayGenre(row, genre), axis=1)

nGenre = len(genre)

genreDict = {}

j = 0

for k in range(nGenre):
    genreDict[genre[k]] = k

    j = k

genreDict['unknown'] = j + 1


def subGenre(row, dizionario):
    if row['genre'] is not None:
        element = row['genre']
        if element in dizionario:
            row['genre'] = genreDict[element]
    return row['genre']


anime['genre'] = anime.apply(lambda row: subGenre(row, genreDict), axis=1)


# type
def creazioneArrayType(row, array):
    if row['type'] is not None:
        array.append(row['type'])


ty = []
anime.apply(lambda row: creazioneArrayType(row, ty), axis=1)

nType = len(ty)

typeDict = {}

j = 0
for n in range(nType):
    typeDict[ty[n]] = n
    j = n
typeDict['unknown'] = j + 1


def subType(row, dizionario):
    if row['type'] is not None:

        element = row['type']

        if element in dizionario:
            row['type'] = typeDict[element]

    return row['type']


anime['type'] = anime.apply(lambda row: subType(row, typeDict), axis=1)


# name
def creazioneArrayTitle(row, array):
    if row['name'] is not None:
        array.append(row['name'])


title = []
anime.apply(lambda row: creazioneArrayTitle(row, title), axis=1)
nTitle = len(title)
titleDict = {}
j = 0
for i in range(nTitle):
    titleDict[title[i]] = i
    j = i
titleDict['unknown'] = j + 1


def subTitle(row, dizionario):
    if row['name'] is not None:
        element = row['name']
        if element in dizionario:
            row['name'] = titleDict[element]
    return row['name']


anime['name'] = anime.apply(lambda row: subTitle(row, titleDict), axis=1)
anime = anime.sort_values('genre')

anime['rating'].fillna(method='ffill', inplace=True)
imputer = KNNImputer(n_neighbors=10, weights="uniform")
anime['rating'] = imputer.fit_transform(anime[['rating']])
anime = anime[anime['episodes'].apply(lambda x: x != 'Unknown')]
anime.reset_index(drop=True)
anime.to_csv('dataset/table.csv', index=False)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSIFICATION
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

target = anime['genre']  # column target
anime.drop(columns=['genre', 'anime_id'], axis=1, inplace=True)

training = anime  # training set
print(training)

x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)

# KNN
knn = KNeighborsClassifier()

# dati da testare
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'hamming']

grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)

grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, error_score=0)
grid_result = grid_search.fit(x_train, y_train)
print("Miglior combinazione di parametri ritrovata:\n")
print(grid_search.best_params_)

y_true, y_pred = y_test, grid_search.predict(x_test)
print(classification_report(y_true, y_pred, target_names=list_genre))
'''
pca = PCA(n_components=3)
pca.fit(animelist)
pcaS = pca.transform(animelist)


ps = pd.DataFrame(pcaS)
print(ps.head())

clusterS = pd.DataFrame(ps[[0,1,2]])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = []
inertia_l = np.empty(8)

# ELBOW METHOD
for i in range(2, 8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(clusterS)
    inertia_l[i] = kmeans.inertia_
    scores.append(silhouette_score(clusterS, kmeans.labels_))

plt.plot(range(0,8),inertia_l,'-o')
plt.xlabel('Number of cluster')
plt.axvline(x=4, color='blue', linestyle='--')
plt.ylabel('Inertia')
plt.show()


from sklearn.cluster import KMeans

clusteredDataset = KMeans(n_clusters=4, random_state=30).fit(clusterS)
centers = clusteredDataset.cluster_centers_
c_preds = clusteredDataset.predict(clusterS)

print(centers)
animelist['cluster'] = c_preds

c0 = animelist[animelist['cluster']==0].drop('cluster',axis=1).mean()
c1 = animelist[animelist['cluster']==1].drop('cluster',axis=1).mean()
c2 = animelist[animelist['cluster']==2].drop('cluster',axis=1).mean()
c3 = animelist[animelist['cluster']==3].drop('cluster',axis=1).mean()

c0.sort_values(ascending=False)[0:15]
'''
