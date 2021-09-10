from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from prediction_genre import *


def reformat_name(row, name):
    for key in name_dic:

        if name_dic[key] == row[name]:
            row[name] = key

            return row[name]


def similarity_with_cosine(row_a, row_b):
    element = cosine_similarity([row_a], [row_b])

    row_a['similarity'] = element[0][0]

    return row_a['similarity']


def set_recommender(episodes, genre, type_, source, duration, name, score):
    cluster_dataset = pd.read_csv('../dataset/anime_preprocessato.csv')

    conversion = pd.read_csv('../dataset/dizionario.csv')

    conversion.apply(lambda row: set_dictionary(row), axis=1)

    cluster_dataset = cluster_dataset.drop(columns=['Producers_format', 'Studios_format'])

    row_user = []

    row_user.append(float(score))

    if episodes == '':
        cluster_dataset = cluster_dataset.drop(columns=['Episodes'])
    else:
        row_user.append(float(episodes))

    if duration == '':
        cluster_dataset = cluster_dataset.drop(columns=['Duration_format'])
    else:
        row_user.append(float(duration))

    row_user.append(float(search_format_genre(genre)))

    if name == '':
        cluster_dataset = cluster_dataset.drop(columns=['Name_format'])
    else:
        row_user.append(float(search_format_name(name)))

    if type_ == '':
        cluster_dataset = cluster_dataset.drop(columns=['Type_format'])
    else:
        row_user.append(float(search_format_name(type_)))

    if source == '':

        cluster_dataset = cluster_dataset.drop(columns=['Source_format'])

    else:
        row_user.append(float(search_format_name(source)))

    # Row_user normalized
    row_user_normalized = preprocessing.normalize([row_user])

    # CLUSTER
    kmeans = KMeans(n_clusters=3).fit(preprocessing.normalize(cluster_dataset))

    cluster_dataset["cluster"] = kmeans.labels_

    # Prediction of row_user's cluster
    prediction = kmeans.predict(row_user_normalized)

    # Split dataset with user cluster
    user_cluster = prediction[0]

    split_cluster = cluster_dataset[cluster_dataset['cluster'].apply(lambda x: x == user_cluster)]

    split_cluster = split_cluster.drop(columns=['cluster'])

    split_cluster['similarity'] = split_cluster.apply(lambda row: similarity_with_cosine(row, row_user), axis=1)

    split_cluster.sort_values(['similarity'], ascending=False, inplace=True)

    set_name()

    ten_sim = split_cluster.head(10)

    ten_sim = ten_sim.loc[:, ['Name_format', 'similarity']]

    ten_sim['Name_format'] = ten_sim.apply(lambda row: reformat_name(row, 'Name_format'),
                                           axis=1)
    ten_sim.sort_values(['similarity'], ascending=False, inplace=True)

    print("Gli anime suggeriti in base alle tue preferenze sono:\n")

    i = 1

    for element in ten_sim['Name_format']:
        print(f"{i}) {element}")
        i = i + 1


def main(episodes, genre, type_, source, duration, name, score):

    set_recommender(episodes, genre, type_, source, duration, name, score)


if __name__ == "__main__":
    main()
