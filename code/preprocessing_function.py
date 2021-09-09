dictionary_genre = {}
conversionDic = {}
dictionary_producers = {}
dictionary_studios = {}
dictionary_rating = {}

genreDizionario = {'Comedy': 0, 'Parody': 1, 'Dementia': 2, 'Kids': 3, 'School': 4, 'Slice of Life': 5, 'Shoujo': 6, 'Romance': 7,
         'Drama': 8, 'Music': 9, 'Fantasy': 10, 'Magic': 11, 'Supernatural': 12, 'Sci-Fi': 13, 'Mecha': 14,
         'Game': 15, 'Sports': 16, 'Shounen': 17, 'Action': 18, 'Adventure': 19, 'Psychological': 20,
         'Mystery': 21, 'Demons': 22, 'Horror': 23, 'Vampire': 24, 'Historical': 25, 'Seinen': 26, 'Ecchi': 27,
         'Hentai': 28, 'Yaoi': 29}


def conversion_string(columnName, columnType, columnProducers, columnStudios, columnSource, columnRating):
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

# Metodo per trasformare i dati categorici in dati numerici
def convert_by_genre(row, column_name):
    if row[column_name] in genreDizionario:
        element = row[column_name]
        row[column_name] = genreDizionario[element]

    return row[column_name]
'''
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

'''


def clean_dataframe(anime):
    anime = anime[anime['Episodes'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Studios'].apply(lambda x: x != 'Unknown')]
    # anime = anime[anime['Score'].apply(lambda x: x != 'Unknown')]
    # anime = anime[anime['Rating'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Genres'].apply(lambda x: x != 'Unknown')]
    anime = anime[anime['Producers'].apply(lambda x: x != 'Unknown')]
    # anime = anime[anime['Duration'].apply(lambda x: x != 'Unknown')]
    # anime = anime[anime['Source'].apply(lambda x: x != 'Unknown')]
    anime.reset_index(drop=True)

    anime["Genres"] = anime["Genres"].replace("Shounen Ai", 'Shounen')
    # anime["Studios"] = anime["Studios"].replace("Unknown", 5).astype(float)
    anime["Score"] = anime["Score"].replace("Unknown", 5).astype(float)
    anime["Rating"] = anime["Rating"].replace("Unknown", 'PG-13 - Teens 13 or older')
    # anime["Genres"] = anime["Genres"].replace("Unknown", )
    # anime["Producers"] = anime["Producers"].replace("Unknown", 5).astype(float)
    anime["Duration"] = anime["Duration"].replace("Unknown", '24 min. per ep.')
    anime["Source"] = anime["Source"].replace("Unknown", 'Manga')
    anime["Source"] = anime["Source"].replace("Web Manga", 'Manga')
    anime["Source"] = anime["Source"].replace("Light novel", 'Novel')
    anime["Source"] = anime["Source"].replace("Visual novel", 'Novel')
    anime["Source"] = anime["Source"].replace("4-koma manga", 'Manga')
    anime["Source"] = anime["Source"].replace("Picture book ", 'Book')
    anime["Source"] = anime["Source"].replace("Card game", 'Game')
    anime["Source"] = anime["Source"].replace("Other", 'Manga')

    return anime


def set_genre_anime(row, column_name):
    list = str(row[column_name]).split(', ')
    row[column_name] = list[0]
    return row[column_name]


'''
def round_rating(row):
    row['rating'] = round(row['rating'], 2)
    row['mean_rating'] = round(row['mean_rating'], 2)
    return row
'''


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


def episodes_duration(row, column):
    if column == 'Episodes':
        element = int(row[column])
        if 1 < element < 10:
            row[column] = 0
        if 10 <= element < 50:
            row[column] = 1
        if element >= 50:
            row[column] = 2
        return row[column]
    else:
        element = int(row[column])
        if 1 < element < 10:
            row[column] = 0
        if 10 <= element < 40:
            row[column] = 1
        if element >= 40:
            row[column] = 2
        return row[column]


def round_rating(row):
    row['Score'] = round(row['Score'])
    return row['Score']
