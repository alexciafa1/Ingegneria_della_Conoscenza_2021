import pandas as pd
from warnings import simplefilter
from preprocessing_function import genre_dictionary
from classification_function import random_forest_classification

simplefilter(action='ignore', category=FutureWarning)

name_dic = {}
conversion_dic = {}
genre_dic = {}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FUNZIONI DI PREDIZIONI DEL GENERE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def set_name():
    i = 0
    for key in conversion_dic:
        if key != 'fine_name' and i == 1:
            name_dic[key] = conversion_dic[key]

        elif key == 'fine_name':
            return
        elif key == 'fine_type':
            i = 1


def set_rating():

    i = 0

    for key in conversion_dic:

        if key != 'fine_name' and i == 1:

            name_dic[key] = conversion_dic[key]

        elif key == 'fine_source':

            return

        elif key == 'fine_rating':
            i = 1


def search_format_name(element):
    if element in conversion_dic:
        return conversion_dic[element]
    else:
        i = 0
        for name in conversion_dic:
            i = i + 1
        conversion_dic[element] = i
        return conversion_dic[element]


def search_format_genre(element):

    if element in genre_dictionary:
        return genre_dictionary[element]



def set_genre():

    for key in conversion_dic:
        if key != 'fine_genre':
            genre_dic[key] = conversion_dic[key]
        else:
            return


def set_dictionary(row):
    conversion_dic[row['Chiave']] = row['Valore']


def predict_genre(name, score, type_, episodes, duration, producers, studios, source):

    anime = pd.read_csv('../dataset/anime_preprocessato.csv')

    conversion = pd.read_csv('../dataset/dizionario.csv')

    conversion.apply(lambda row: set_dictionary(row), axis=1)

    target = anime['Genre_format']  # column target

    training = anime.drop(columns=['Genre_format'])  # training set

    name_format = search_format_name(name)

    type_format = search_format_name(type_)

    producers_format = search_format_name(producers)

    studios_format = search_format_name(studios)

    source_format = search_format_name(source)

    row_user = [score, episodes, duration, producers_format, studios_format, name_format, type_format, source_format]

    rf = random_forest_classification(training, target)

    genre_predict = rf.predict([row_user])

    final_val = [key for key, val in genre_dictionary.items() if val == genre_predict[0]]

    print(f"Il genere dell'anime {name} e': ", final_val[0])

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FINE FUNZIONI DI PREDIZIONI DEL GENERE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++