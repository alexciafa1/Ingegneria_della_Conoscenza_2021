from sklearn import preprocessing
from classification_function import *
from analysys.preprocessing_function import *

anime = pd.DataFrame()

data = pd.read_csv('../dataset/anime.csv')

data = data.iloc[:, [1, 2, 3, 6, 7, 10, 12, 13, 14]]

anime = data.copy()

anime = clean_dataframe(anime)

# format Duration
anime['Duration_format'] = anime.apply(lambda row: format_duration_anime(row), axis=1)

# format genre
anime.apply(lambda row: creation_frequency_dictionary(row, dictionary_frequency_genre, 'Genres'), axis=1)

anime['Genre_format'] = anime.apply(lambda row: set_columns_anime(row, dictionary_frequency_genre, 'Genres'), axis=1)

anime.sort_values(['Genre_format'], ascending=True, inplace=True)

# format producers
anime.apply(lambda row: creation_frequency_dictionary(row, producers_dictionary, 'Producers'), axis=1)

anime['Producers_format'] = anime.apply(lambda row: set_columns_anime(row, producers_dictionary, 'Producers'), axis=1)

# format Studios
anime.apply(lambda row: creation_frequency_dictionary(row, studios_dictionary, 'Studios'), axis=1)

anime['Studios_format'] = anime.apply(lambda row: set_columns_anime(row, studios_dictionary, 'Studios'), axis=1)

# array di conversione per categorici/numerici
anime.apply(
    lambda column: conversion_string(anime['Name'], anime['Type'], anime['Producers_format'],
                                     anime['Studios_format'], anime['Source']), axis=0)


conversionDataset = pd.DataFrame()

conversionDataset['Chiave'] = conversion_dictionary.keys()

conversionDataset['Valore'] = conversion_dictionary.values()

conversionDataset.to_csv("../dataset/dizionario.csv", index=False)

# name from categorical to numeric
anime['Name_format'] = anime.apply(lambda row: convert_by_column(row, 'Name'), axis=1)

# genre from categorical to numeric
anime['Genre_format'] = anime.apply(lambda row: convert_by_genre(row, 'Genre_format'), axis=1)

# producers from categorical to numeric
anime['Producers_format'] = anime.apply(lambda row: convert_by_column(row, 'Producers_format'), axis=1)

# studios from categorical to numeric
anime['Studios_format'] = anime.apply(lambda row: convert_by_column(row, 'Studios_format'), axis=1)

# type from categorical to numeric
anime['Type_format'] = anime.apply(lambda row: convert_by_column(row, 'Type'), axis=1)

# Studios from categorical to numeric
anime['Source_format'] = anime.apply(lambda row: convert_by_column(row, 'Source'), axis=1)

anime_pp = anime.drop(columns=['Name', 'Genres', 'Type', 'Producers', 'Source', 'Duration', 'Studios'])

anime_pp.to_csv("../dataset/anime_preprocessato.csv", index=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSIFICATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

target = anime_pp['Genre_format']  # column target

training = anime_pp.drop(columns=['Genre_format'])  # training set

# Create the Scaler object and fit your data on the scaler object
scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(training)

training = pd.DataFrame(scaled_df)

# KNN CLASSIFICATION
knn = knn_classification(training, target)

# GAUSSIAN CLASSIFICATION
gaussian_nb_classification(training, target)

# RANDOM FOREST CLASSIFICATION
random_forest = random_forest_classification(training, target)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# END CLASSIFICATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
