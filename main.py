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

from sklearn import model_selection, metrics
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

from rating_processing import merge_with_rating


genre_ ='genre'
name_ = 'name'
type_ = 'type'

array_genre = []
array_type = []
array_name = []
new_genre = []
list_genre = []
target_names = []

genreDict = {}
typeDict = {}
titleDict = {}
frequency_genre = {}

df = pd.read_csv('dataset/anime.csv')

df.drop('members', axis=1, inplace=True)

anime = merge_with_rating(df)

# anime.to_csv('dataset/mergeDataset.csv', index=False)



def creationFrequency_genre(row, dictionary):
    list_ = str(row['genre']).split(', ')
    for genre_ in list_:
        if genre_ not in frequency_genre:
            dictionary[genre_] = 1
        else:
            dictionary[genre_] = dictionary[genre_] + 1


def set_target_names(genre):
    if genre not in target_names:
        target_names.append(genre)


def set_genre_anime(row):
    list_ = str(row['genre']).split(', ')
    genre_frequency = 0
    final_genre = ''
    for genre in list_:
        if frequency_genre[genre] > genre_frequency:
            genre_frequency = frequency_genre[genre]
            final_genre = genre
    new_genre.append(final_genre)


'''
def creazionelistGenre(row, list_genre):
    list_ = str(row['genre']).split(', ')
    for genre_ in list_:
        if genre_ not in list_genre:
            list_genre.append(genre_)
'''

anime.apply(lambda row: creationFrequency_genre(row, frequency_genre), axis=1)
print(frequency_genre)
anime.apply(lambda row: set_genre_anime(row), axis=1)
anime['genre'] = new_genre

#

anime['genre'].to_csv("dataset/final_genre.csv", index=False)


def creazionelistGenre(row, list_genre):
    if row['genre'] not in list_genre:
        list_genre.append(row['genre'])


df = pd.DataFrame()
anime.apply(lambda row: creazionelistGenre(row, list_genre), axis=1)

df['lista generi'] = list_genre
df['lista generi'].to_csv('dataset/lista generi.csv', index=False)


# Metodo per trasformare i dati categorici in dati numerici
def subByColumn(row, dizionario, column_name):
    if row[column_name] is not None:
        element = row[column_name]
        if element in dizionario:
            row[column_name] = dizionario[element]
    return row[column_name]


# Method for creation array (name, type, genre, ...)
def creazioneArrayByColumn(row, array, column_name):
    if row[column_name] is not None:
        array.append(row[column_name])


def create_dictionary(array, dizionario):
    size_array = len(array)
    j = 0
    for k in range(size_array):
        dizionario[array[k]] = k

        j = k
    dizionario['unknown'] = j + 1


anime.apply(lambda row: set_target_names(row['genre']), axis=1)
# target_names.remove('nan')

# print("target_names: \n", target_names)
# print("grandezza target_names: ", len(target_names))

# genre
anime.apply(lambda row: creazioneArrayByColumn(row, array_genre, genre_), axis=1)

create_dictionary(array_genre, genreDict)

anime['genre'] = anime.apply(lambda row: subByColumn(row, genreDict, genre_), axis=1)

# type
anime.apply(lambda row: creazioneArrayByColumn(row, array_type, type_), axis=1)

create_dictionary(array_type, typeDict)

anime['type'] = anime.apply(lambda row: subByColumn(row, typeDict, type_), axis=1)

# name anime
anime.apply(lambda row: creazioneArrayByColumn(row, array_name, name_), axis=1)

create_dictionary(array_name, titleDict)

anime['name'] = anime.apply(lambda row: subByColumn(row, titleDict, name_), axis=1)

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
knn = KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='uniform')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("Accuracy knn:", metrics.accuracy_score(y_test, y_pred))

# GAUSSIAN
gau = GaussianNB()
gau.fit(x_train, y_train)
y_pred_gau = gau.predict(x_test)
print("Accuracy gau :", metrics.accuracy_score(y_test, y_pred_gau))
# RANDOM FOREST
rf = RandomForestClassifier(n_estimators=1000, criterion='gini')
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print("Accuracy rf:", metrics.accuracy_score(y_test, y_pred_rf))

'''
# dati da testare
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'hamming']

grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=grid, cv=cv, n_jobs=-1, error_score=0)
grid_result = grid_search.fit(x_train, y_train)
print("Miglior combinazione di parametri ritrovata:\n")
print(grid_search.best_params_)

y_true, y_pred = y_test, grid_search.predict(x_test)
# print('y_pred;\n', y_pred, '\ny_true;\n', y_true)
print(classification_report(y_true, y_pred, target_names=target_names))
'''
'''
#---------------------------------------------------------------------

y_test.apply(lambda row: set_target_names(row))
# target_names.remove('nan')
print("target_names: \n", target_names)
print("grandezza target_names: ", len(target_names))
print('Target\n', target)
#---------------------------------------------------------------------
'''

'''
Kfold = model_selection.KFold(n_splits=10, random_state=None)
knn_model = KNeighborsClassifier(metric='hamming', n_neighbors= 3, weights= 'distance')

scoring = {'accuracy':make_scorer(accuracy_score),
            'precision':make_scorer(precision_score, average='macro',zero_division=0),
            'recall':make_scorer(recall_score, average='macro',zero_division=0)}

knn = cross_validate(knn_model, training, target, cv=Kfold, scoring=scoring)

models_scores_table = pd.DataFrame( {'KNearestNeighbor':[knn['test_accuracy'].mean(),
                                                              knn['test_precision'].mean(),
                                                              knn['test_recall'].mean()]},

                                      index=['Accuracy', 'Precision', 'Recall'])

models_scores_table.to_csv("dataset/models_scores_table.csv", index=False)
'''
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
