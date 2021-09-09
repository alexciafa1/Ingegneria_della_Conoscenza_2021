import os

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from classification_function import *
from preprocessing_function import *

anime = pd.DataFrame()
data = pd.read_csv('../dataset/anime.csv')
data = data.iloc[:, [1, 2, 3, 6, 7, 15, 10, 12, 13, 14]]
anime = data.copy()
dictionary_genre_format={}

anime = clean_dataframe(anime)

# format Duration
anime['Duration_format'] = anime.apply(lambda row: format_duration_anime(row), axis=1)

# format rating
anime['Rating_format'] = anime.apply(lambda row: set_rating(row), axis=1)
anime.apply(lambda row: creation_frequency_dictionary(row, dictionary_rating, 'Rating_format'), axis=1)
# print("dictionary_rating ", dictionary_rating)


# format genre
anime.apply(lambda row: creation_frequency_dictionary(row, dictionary_genre, 'Genres'), axis=1)
# print("frequency ", dictionary_genre)
anime['Genre_format'] = anime.apply(lambda row: set_columns_anime(row, dictionary_genre, 'Genres'), axis=1)
print("leng dic_genre ", len(dictionary_genre))

# format producers
anime.apply(lambda row: creation_frequency_dictionary(row, dictionary_producers, 'Producers'), axis=1)
# print("frequency ", dictionary_producers)
anime['Producers_format'] = anime.apply(lambda row: set_columns_anime(row, dictionary_producers, 'Producers'), axis=1)

# format Studios
anime.apply(lambda row: creation_frequency_dictionary(row, dictionary_studios, 'Studios'), axis=1)
# print("frequency ", dictionary_studios)
anime['Studios_format'] = anime.apply(lambda row: set_columns_anime(row, dictionary_studios, 'Studios'), axis=1)
anime.to_csv("../dataset/anime_ridotto_2.csv", index=False)

# array di conversione per categorici/numerici
anime.apply(
    lambda column: conversion_string(anime['Genre_format'], anime['Name'], anime['Type'], anime['Producers_format'],
                                     anime['Studios_format'], anime['Source'], anime['Rating_format']), axis=0)

conversionDataset = pd.DataFrame()
# conversionDataset.columns=['Chiave', 'Valore']

conversionDataset['Chiave'] = conversionDic.keys()
conversionDataset['Valore'] = conversionDic.values()

conversionDataset.to_csv("../dataset/conversion_dic.csv", index=False)

# name from categorical to numeric
anime['Name_format'] = anime.apply(lambda row: convert_by_column(row, 'Name'), axis=1)

# genre from categorical to numeric
anime['Genre_format'] = anime.apply(lambda row: convert_by_column(row, 'Genre_format'), axis=1)

# producers from categorical to numeric
anime['Producers_format'] = anime.apply(lambda row: convert_by_column(row, 'Producers_format'), axis=1)

# studios from categorical to numeric
anime['Studios_format'] = anime.apply(lambda row: convert_by_column(row, 'Studios_format'), axis=1)

# type from categorical to numeric
anime['Type_format'] = anime.apply(lambda row: convert_by_column(row, 'Type'), axis=1)

# Studios from categorical to numeric
anime['Source_format'] = anime.apply(lambda row: convert_by_column(row, 'Source'), axis=1)

# Rating from categorical to numeric
anime['Rating_format'] = anime.apply(lambda row: convert_by_column(row, 'Rating_format'), axis=1)

anime['Score'] = anime.apply(lambda row: round_rating(row), axis=1)


#anime['Episodes'] = anime.apply(lambda row: episodes_duration(row, 'Episodes'), axis=1)

#anime['Duration_format'] = anime.apply(lambda row: episodes_duration(row, 'Duration_format'), axis=1)

anime_pp = anime.drop(columns=['Name', 'Genres', 'Type', 'Producers', 'Source', 'Duration', 'Studios'])

anime_pp.to_csv("../dataset/anime_pp_2.csv", index=False)

os.system("pause")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSIFICATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

target = anime_pp['Genre_format']  # column target
training = anime_pp.drop(columns=['Genre_format', 'Rating'])  # training set
# print(training)

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
'''
# KNN CLASSIFICATION
knn = KNNClassification(training, target)

# CROSS VALIDATE
Kfold = model_selection.KFold(n_splits=10, random_state=None)
# evaluate_with_crossvalidate(knn, x_train, y_train, Kfold)

# GAUSSIAN CLASSIFICATION
GaussianNBClassification(training, target)

# RANDOM FOREST CLASSIFICATION
random_forest = RandomForestClassifierClassification(training, target)
# evaluate_with_crossvalidaterf(random_forest, x_train, y_train, Kfold)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# END CLASSIFICATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
