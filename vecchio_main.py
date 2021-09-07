import os

import pandas as pd
from sklearn import preprocessing

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from code.rating_processing import merge_with_rating
from code.preprocessing_function import *

genre_ = 'genre_1'
name_ = 'name'
type_ = 'type'

array_episodes = []
array_genre = []
array_type = []
array_name = []
# new_genre = []
list_genre = []
# target_names = []

genreDict = {}
typeDict = {}
titleDict = {}
episodesDict ={}
# frequency_genre = {}

anime = pd.DataFrame()
data = pd.read_csv('dataset/anime.csv')
data.drop('members', axis=1, inplace=True)
anime = data.copy()

anime = clean_dataframe(anime)
print(len(anime))

anime.to_csv("anime_pre.csv", index=False)

anime = merge_with_rating(anime)
anime = anime.apply(lambda row: durata_episodes(row), axis=1)
print(anime['episodes'])
# anime = anime.apply(lambda row: round_rating(row), axis=1)
# print(anime['mean_rating'],"\n", anime['rating'] )
# anime.loc[anime.mean_rating <= 0, 'mean_rating'] = 5
os.system("pause")
anime.to_csv("anime_merge.csv", index=False)



# anime.to_csv('dataset/mergeDataset.csv', index=False)

anime.apply(lambda row: creation_frequency_dictionary(row, frequency_genre), axis=1)
# print(frequency_genre)
anime['genre'] = anime.apply(lambda row: set_columns_anime(row), axis=1)

anime['genre_1'] = new_genre

# print("leng new_genre ", len(new_genre))

anime.to_csv("dataset/final_genre.csv", index=False)

anime.apply(lambda column: conversion_string(anime['genre'], anime['type'], anime['name'], anime['episodes']), axis=0)

# print("conversion: \n", conversionDic)

anime.apply(lambda row: creation_list_genre(row, list_genre), axis=1)

# print("list genre ", list_genre)

# anime['lista generi'] = list_genre

# anime['lista generi'].to_csv('dataset/lista_generi_diminuita.csv', index=False)

# anime.apply(lambda row: set_target_names(row['genre']), axis=1)
# target_names.remove('nan')

# print("target_names: \n", target_names)
# print("grandezza target_names: ", len(target_names))

# genre
anime.apply(lambda row: creation_array_by_column(row, array_genre, genre_), axis=1)

create_dictionary(array_genre, genreDict)

anime[genre_] = anime.apply(lambda row: sub_by_column(row, genreDict, genre_), axis=1)

# type
anime.apply(lambda row: creation_array_by_column(row, array_type, type_), axis=1)

create_dictionary(array_type, typeDict)

anime[type_] = anime.apply(lambda row: sub_by_column(row, typeDict, type_), axis=1)

# name anime
anime.apply(lambda row: creation_array_by_column(row, array_name, name_), axis=1)

create_dictionary(array_name, titleDict)

anime[name_] = anime.apply(lambda row: sub_by_column(row, titleDict, name_), axis=1)

# episodes
# genre
anime.apply(lambda row: creation_array_by_column(row, array_episodes, 'episodes'), axis=1)

create_dictionary(array_episodes, episodesDict)

anime['episodes'] = anime.apply(lambda row: sub_by_column(row, episodesDict, 'episodes'), axis=1)
# anime = anime.sort_values('genre')

# anime['rating'].fillna(method='ffill', inplace=True)

anime = anime[anime['rating'].apply(lambda x: not isnull(x))]

anime = anime[anime['mean_rating'].apply(lambda x: x >= 0)]



anime.reset_index(drop=True)


# anime = anime[anime['episodes'].apply(lambda x: x != 'Unknown')]
# anime.reset_index(drop=True)

# anime = anime.drop(columns=['genre'])

print("frequency genre \n", frequency_genre)
print("dictionary \n", genreDict)
anime.to_csv('dataset/table.csv', index=False)

os.system("pause")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSIFICATION
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

target = anime['genre_1']  # column target

anime.drop(columns=['genre', 'genre_1', 'anime_id', 'rating'], axis=1, inplace=True)

training = anime  # training set
'''
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(training)
training = pd.DataFrame(scaled_df)
'''

x = training.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
training = pd.DataFrame(x_scaled)

print(training)

x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)


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
