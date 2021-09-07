from pandas import isnull
from sklearn.impute import KNNImputer

new_genre = []
target_names = []
dictionary_genre = {}
conversionDic = {}


def conversion_string(columnGenre, columnName, columnType, columnProducers, columnStudios, columnSource):
    i = 0
    for genre in columnGenre:
        if genre not in conversionDic:
            conversionDic[genre] = i
            i = i + 1
    conversionDic["fine_genre"]=-1
    i = 0
    for type in columnType:
        if type not in conversionDic:
            conversionDic[type] = i
            i = i + 1
    conversionDic["fine_type"] = -1
    i = 0
    for name in columnName:
        if name not in conversionDic:
            conversionDic[name] = i
            i = i + 1
    conversionDic["fine_name"] = -1
    i = 0
    for producers in columnProducers:
        if producers not in conversionDic:
            conversionDic[producers] = i
            i = i + 1
    conversionDic["fine_producers"] = -1
    i = 0
    for studios in columnStudios:
        if studios not in conversionDic:
            conversionDic[studios] = i
            i = i + 1
    conversionDic["fine_studios"] = -1
    i = 0
    for source in columnSource:
        if source not in conversionDic:
            conversionDic[source] = i
            i = i + 1
    conversionDic["fine_source"] = -1


def creation_frequency_dictionary(row, dictionary, column_name):
    list_ = str(row[column_name]).split(', ')

    for genre_ in list_:
        if genre_ not in dictionary:
            dictionary[genre_] = 1
        else:
            dictionary[genre_] = dictionary[genre_] + 1


def set_target_names(genre):
    if genre not in target_names:
        target_names.append(genre)


def set_columns_anime(row, dictionary, column_name):
    list_ = str(row[column_name]).split(', ')
    genre_frequency = 0
    final_genre = ''
    for name in list_:
        if dictionary[name] > genre_frequency:
            genre_frequency = dictionary[name]
            final_genre = name
    row['genre'] = final_genre
    new_genre.append(final_genre)
    return row['genre']


'''
def creazionelistGenre(row, list_genre):
    if row['genre'] not in list_genre:
        list_genre.append(row['genre'])
'''


def creation_list_genre(row, list_genre):
    list_ = str(row['genre']).split(', ')
    for genre_ in list_:
        if genre_ not in list_genre:
            list_genre.append(genre_)


# Metodo per trasformare i dati categorici in dati numerici
def sub_by_column(row, column_name):
    if row[column_name] in conversionDic:
        element = row[column_name]
        row[column_name] = conversionDic[element]

    return row[column_name]


# Method for creation array (name, type, genre, ...)
def creation_array_by_column(row, array, column_name):
    if row[column_name] is not None:
        array.append(row[column_name])


def create_dictionary(array, dizionario):
    size_array = len(array)
    j = 0
    for k in range(size_array):
        dizionario[array[k]] = k

    j = k
    dizionario['unknown'] = j + 1


def clean_dataframe(anime):
    anime = anime[anime['genre'].apply(lambda x: not isnull(x))]
    anime.reset_index(drop=True)

    anime = anime[anime['type'].apply(lambda x: not isnull(x))]
    anime.reset_index(drop=True)

    anime = anime[anime['episodes'].apply(lambda x: x != 'Unknown')]
    anime.reset_index(drop=True)

    imputer = KNNImputer(n_neighbors=10, weights="uniform")
    anime['rating'] = imputer.fit_transform(anime[['rating']])

    return anime


def round_rating(row):
    row['rating'] = round(row['rating'], 2)
    row['mean_rating'] = round(row['mean_rating'], 2)
    return row


def durata_episodes(row):
    durata = int(row['episodes'])
    if 1 <= durata <= 10:
        new = 'breve'
    if 11 <= durata <= 50:
        new = 'medio'
    if durata > 50:
        new = 'lungo'
    row['episodes'] = new
    return row


def format_duration_anime(row):
    list_ = str(row['Duration']).split(' ')
    if 'per' in list_:
        list_.remove('per')
        list_.remove('ep.')
    if 'hr.' in list_:
        list_.remove('hr.')
    if 'min.' in list_:
        list_.remove('min.')
    i = 0
    new = 0
    if len(list_) > 1:
        for element in list_:
            if i==0:
                new = int(element)*60
                i = i + 1
            else:
                new = new + int(element)
    else:
        new = list_[0]
    row['Duration_format'] = new
    return row['Duration_format']







    #print(row['Duration'])
   # for element in list_:
       # if 'hr' in element:


    #print(list_)
