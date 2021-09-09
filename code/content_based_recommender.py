import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from predictgenre import search_format_name, set_dictionary, set_genre, genre_dic, set_name, name_dic, set_rating, \
    search_format_genre


def reformat_name(row, name):
    for key in name_dic:
        if name_dic[key] == row[name]:
            row[name] = key
            return row[name]


def similarity_with_cosine(row_a, row_b):
    a = np.array(row_a, dtype=float)
    b = np.array(row_b, dtype=float)
    element = cosine_similarity([a], [b])
    row_a['similarity'] = element[0][0]
    return row_a['similarity']


def set_recommender(episodes, genre, type_, source, duration, name, score):
    '''
    richiamare il cluster
    predire il cluster
    splittatre sul cluster
    similarità (coseno) sul split
    restituire i primi 10 argomenti
    '''

    cluster_dataset = pd.read_csv('../dataset/anime_pp_2.csv')

    cluster_dataset = cluster_dataset.drop(columns=['Rating', 'Rating_format', 'Producers_format', 'Studios_format'])

    conversion = pd.read_csv('../dataset/conversion_dic.csv')

    conversion.apply(lambda row: set_dictionary(row), axis=1)
    set_genre()
    set_rating()
    #print("genre dic\n", genre_dic)
    #rating_format = search_format_name(rating)
    genre_format = search_format_genre(genre)
    name_format = search_format_name(name)
    type_format = search_format_name(type_)
    #producers_format = search_format_name(producers)
   # studios_format = search_format_name(studio)
    source_format = search_format_name(source)

    #k = 3
    # Create the Scaler object
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(cluster_dataset)
    #cluster_dataset_scaled = pd.DataFrame(x_scaled)

    kmeans = KMeans(n_clusters=3).fit(preprocessing.normalize(cluster_dataset))
    # Applying kmeans to the dataset / Creating the kmeans classifier
    #kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #y_kmeans = kmeans.fit_predict(cluster_dataset_scaled)
    # print("label del y_means", kmeans.labels_)
    cluster_dataset["cluster"] = kmeans.labels_
    cluster_dataset.to_csv("cluster_rec.csv", index=False)
    # print(cluster_dataset['cluster'])
    # cluster_dataset.to_csv("cluster.csv", index=False)

    row_utente = [score, episodes, duration, genre_format, name_format,
                  type_format, source_format]

    row_utente_scaled = preprocessing.normalize([row_utente])

    clust_pre = kmeans.predict(row_utente_scaled)

    # final_val = [key for key, val in genre_dic.items() if val == clust_pre[0]]
    print("Il cluster dell'anime e': ", clust_pre[0])
    number_cluster = clust_pre[0]
    split = cluster_dataset[cluster_dataset['cluster'].apply(lambda x: x == number_cluster)]
    split2 = cluster_dataset[cluster_dataset['cluster'].apply(lambda x: x == 0)]
    split3 = cluster_dataset[cluster_dataset['cluster'].apply(lambda x: x == 2)]
    split.to_csv("split.csv", index=False)
    split2.to_csv("split2.csv", index=False)
    split3.to_csv("split3.csv", index=False)

    # Similitudine tupla

    similarity_dataset = split.drop(columns=['cluster'])

    similarity_dataset2 = split2.drop(columns=['cluster'])

    similarity_dataset3 = split3.drop(columns=['cluster'])

    similarity_dataset['similarity'] = similarity_dataset.apply(lambda row: similarity_with_cosine(row, row_utente),
                                                                axis=1)
    similarity_dataset2['similarity'] = similarity_dataset2.apply(lambda row: similarity_with_cosine(row, row_utente),
                                                                axis=1)
    similarity_dataset3['similarity'] = similarity_dataset3.apply(lambda row: similarity_with_cosine(row, row_utente),
                                                                  axis=1)
    similarity_dataset.sort_values(['similarity'], ascending=False, inplace=True)

    similarity_dataset2.sort_values(['similarity'], ascending=False, inplace=True)

    similarity_dataset3.sort_values(['similarity'], ascending=False, inplace=True)
    similarity_dataset.to_csv("top_sim.csv", index=False)
    similarity_dataset2.to_csv("top_sim2.csv", index=False)
    similarity_dataset3.to_csv("top_sim3.csv", index=False)
    set_name()

   # print("name: ", name_dic)

    first_10_similarity = similarity_dataset.head(10)
    first_10_similarity = first_10_similarity.iloc[:, [4]]

    #first_10_similarity.apply(lambda row: reformat_name(row))
    print("row_utente\n" , row_utente)
    #print("nm: ", name_dic)
    first_10_similarity['Name_format'] = first_10_similarity.apply(lambda row: reformat_name(row, 'Name_format'), axis=1)
    print("similarity:\n ", first_10_similarity)
