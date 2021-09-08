import csv
import os

import pandas as pd
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from classification_function import KNNClassification
name_dic = {}

conversion_dic = {}
genre_dic = {}
def set_name():
    i = 0
    for key in conversion_dic:
        if key!='fine_name' and i == 1:
            name_dic[key] = conversion_dic[key]

        elif key=='fine_name':
            return
        elif key=='fine_type':
            i = 1
def set_rating():
    i = 0
    for key in conversion_dic:
        if key!='fine_name' and i == 1:
            name_dic[key] = conversion_dic[key]

        elif key=='fine_source':
            return
        elif key=='fine_rating':
            i = 1

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
    genre_predict = knn.predict([[score, episodes, duration, producers_format, studios_format, name_format, type_format, source_format]])
    final_val = [key for key, val in genre_dic.items() if val == genre_predict[0]]
    print(f"Il genere dell'anime {name} e': ", final_val[0])

    print("Ecco a lei senpai-sama")
