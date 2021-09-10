import pandas as pd

dictionary_genre_format = {}
dictionary_frequency_genre = {}
conversion_dictionary = {}
producers_dictionary = {}
studios_dictionary = {}
rating_dictionary = {}
genre_dictionary = {'comedy': 0, 'parody': 1, 'dementia': 2, 'kids': 3, 'school': 4, 'slice of life': 5, 'shoujo': 6,
                    'romance': 7,
                    'drama': 8, 'music': 9, 'fantasy': 10, 'magic': 11, 'supernatural': 2, 'sci-fi': 13, 'mecha': 14,
                    'game': 15, 'sports': 16, 'shounen': 17, 'action': 18, 'adventure': 19, 'psychological': 20,
                    'mystery': 21, 'demons': 22, 'horror': 23, 'vampire': 24, 'historical': 25, 'seinen': 26,
                    'ecchi': 27,
                    'hentai': 28, 'yaoi': 29}


def conversion_string(column_name, column_type, column_producers, column_studios, column_source):
    i = 0
    for type_ in column_type:
        if type_ not in conversion_dictionary:
            conversion_dictionary[type_] = i
            i = i + 1
    conversion_dictionary["fine_type"] = -1
    i = 0
    for name in column_name:
        if name not in conversion_dictionary:
            conversion_dictionary[name] = i
            i = i + 1
    conversion_dictionary["fine_name"] = -1
    i = 0
    for producers in column_producers:
        if producers not in conversion_dictionary:
            conversion_dictionary[producers] = i
            i = i + 1
    conversion_dictionary["fine_producers"] = -1
    i = 0
    for studios in column_studios:
        if studios not in conversion_dictionary:
            conversion_dictionary[studios] = i
            i = i + 1
    conversion_dictionary["fine_studios"] = -1
    i = 0
    for source in column_source:
        if source not in conversion_dictionary:
            conversion_dictionary[source] = i
            i = i + 1
    conversion_dictionary["fine_source"] = -1


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
    if row[column_name] in conversion_dictionary:
        element = row[column_name]
        row[column_name] = conversion_dictionary[element]

    return row[column_name]


# Metodo per trasformare i dati categorici in dati numerici
def convert_by_genre(row, column_name):
    if row[column_name] in genre_dictionary:
        element = row[column_name]
        row[column_name] = genre_dictionary[element]

    return row[column_name]


def clean_dataframe(anime):
    anime = anime[anime['Episodes'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Studios'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Type'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Genres'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Producers'].apply(lambda x: x != 'Unknown')]
    anime.reset_index(drop=True)
    anime["Genres"] = anime["Genres"].replace("Shounen Ai", 'Shounen')
    anime["Score"] = anime["Score"].replace("Unknown", 5).astype(float)
    anime["Duration"] = anime["Duration"].replace("Unknown", '24 min. per ep.')
    anime["Source"] = anime["Source"].replace("Unknown", 'Manga')
    anime["Source"] = anime["Source"].replace("Web Manga", 'Manga')
    anime["Source"] = anime["Source"].replace("Light novel", 'Novel')
    anime["Source"] = anime["Source"].replace("Visual novel", 'Novel')
    anime["Source"] = anime["Source"].replace("4-koma manga", 'Manga')
    anime["Source"] = anime["Source"].replace("Picture book ", 'Book')
    anime["Source"] = anime["Source"].replace("Card game", 'Game')
    anime["Source"] = anime["Source"].replace("Other", 'Manga')
    anime = anime.applymap(lambda x: str(x).lower() if pd.notnull(x) else x)
    return anime


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
                new = int(element) * 60
                i = i + 1
            else:
                new = new + int(element)
    else:
        new = list_[0]
    row['Duration_format'] = new
    return row['Duration_format']
