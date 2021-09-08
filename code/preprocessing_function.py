from pandas import isnull
from sklearn.impute import KNNImputer

dictionary_genre = {}
conversionDic = {}
dictionary_producers = {}
dictionary_studios = {}
dictionary_rating = {}


def conversion_string(columnGenre, columnName, columnType, columnProducers, columnStudios, columnSource, columnRating):
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
    i = 0
    for rating in columnRating:
        if rating not in conversionDic:
            conversionDic[rating] = i
            i = i + 1
    conversionDic["fine_rating"] = -1


def set_rating(row):
    if row['Rating'] == 'R - 17+ (violence & profanity)':
        row['Rating'] = 'R'
    if row['Rating'] == 'PG-13 - Teens 13 or older':
        row['Rating'] = 'PG-13'
    if row['Rating'] == 'PG - Children':
        row['Rating'] = 'PG-13'
    if row['Rating'] == 'R+ - Mild Nudity':
        row['Rating'] = 'R+'
    if row['Rating'] == 'G - All Ages':
        row['Rating'] = 'G'
    if row['Rating'] == 'Rx - Hentai':
        row['Rating'] = 'R+'
    return row['Rating']


def creation_frequency_dictionary(row, dictionary, column_name):
    list_ = str(row[column_name]).split(', ')
    for element in list_:
        if element not in dictionary:
            dictionary[element] = 1
        else:
            dictionary[element] = dictionary[element] + 1


def set_columns_anime(row, dictionary, column_name):
    list_ = str(row[column_name]).split(', ')
    genre_frequency = 0
    final_genre = ''
    for name in list_:
        if dictionary[name] > genre_frequency:
            genre_frequency = dictionary[name]
            final_genre = name
    row[column_name] = final_genre
    return row[column_name]


# Metodo per trasformare i dati categorici in dati numerici
def convert_by_column(row, column_name):
    if row[column_name] in conversionDic:
        element = row[column_name]
        row[column_name] = conversionDic[element]

    return row[column_name]


def clean_dataframe(anime):
    anime = anime[anime['Episodes'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Studios'].apply(lambda x: x != 'Unknown')]
    anime["Score"] = anime["Score"].replace("Unknown", 5).astype(float)
    # anime = anime[anime['Score'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Rating'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Genres'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Producers'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Duration'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Source'].apply(lambda x: x != 'Unknown')]
    anime.reset_index(drop=True)
    return anime


def round_rating(row):
    row['rating'] = round(row['rating'], 2)
    row['mean_rating'] = round(row['mean_rating'], 2)
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
            if i == 0:
                new = int(element)*60
                i = i + 1
            else:
                new = new + int(element)
    else:
        new = list_[0]
    row['Duration_format'] = new
    return row['Duration_format']
