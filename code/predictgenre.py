import csv
import os

import pandas as pd

from classification_function import KNNClassification
from preprocessing_function import conversionDic

conversion_dic = {}
genre_dic = {}


def search_format_name(element):
    return conversion_dic[element]


def set_genre():
    for key in conversion_dic:
        if key != 'fine_genre':
            genre_dic[key] = conversion_dic[key]
        else:
            return


def set_dictionary(row):
    conversion_dic[row['Chiave']] = row['Valore']


def predict_genre(name, score, type_, episodes, duration, producers, studios, source):
    anime = pd.read_csv('../dataset/anime_pp.csv')
    conversion = pd.read_csv('../dataset/conversion_dic.csv')

    conversion.apply(lambda row: set_dictionary(row), axis=1)
    set_genre()
    print("genre dic\n", genre_dic)
    target = anime['Genre_format']  # column target
    training = anime.drop(columns=['Genre_format'])  # training set

    name_format = search_format_name(name)
    type_format = search_format_name(type_)
    producers_format = search_format_name(producers)
    studios_format = search_format_name(studios)
    source_format = search_format_name(source)

    knn = KNNClassification(training, target)

    predizione_genere = knn.predict([name_format, score, type_format, episodes, duration, producers_format, studios_format, source_format])
    print(" numero genere ",predizione_genere)
     # genre_predict =

    print("ecco a lei senpai-sama")
