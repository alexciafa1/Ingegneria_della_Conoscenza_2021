from predictgenre import predict_genre
from recommender import set_recommender

print("^^^^^^^^^^^^^^^^^^^ ANIME RECOMMENDER ^^^^^^^^^^^^^^^^^^^\n")
string = "Konichiwa, cosa vuoi fare?\n" + "Vuoi che ti suggerisca un anime ? - premi 1\n" \
         + "Vuoi sapere il genere di un anime? - premi 2\n" \
         + "Vuoi uscire? - premi 3\n"

risposta = input(string)

if risposta == '1':
    name = ''
    score = ''
    genre = ''
    type_ = ''
    duration = ''
    episodes = ''
    source = ''
    bol = False
    print(" ti faro' delle domande per poterti suggerire un anime.")

    name = input(" qual e' questo anime?\n").lower()

    score = input("Se dovessi valutarlo che voto gli daresti da 1 a 10?\n").lower()

    genre = input("Qual e' il genere di questo anime?\n").lower()

    string = input("Sai dirmi il tipo di questo anime? (Si o  No)(se non hsai cosa è il tipo digita 1)\n").lower()

    while string == '1':
        print("Qaundo parliamo di tipo intendiamo se l'anime appartiene a queste categorie:\n" +
              "-TV : l'anime viene trasmesso ad episodi\n" +
              "-Movie: l'anime è un film\n" +
              "-ONA/OVA : l'anime è una puntata speciale di un anime o originale")

        string = input("Ora sai dirmi qual e' il tipo?(Si o  No)(se non hsai cosa è il tipo digita 1)\n").lower()

    if string == "si":
        type_ = input("Qual e' il tipo tra TV, Original...?\n").lower()

    if type_ == 'tv' or type_.lower() == 'ona' or type_.lower() == 'ova':

        string = input("Sai dirmi da quanti episodi è formato? (si o scrivi qualcos per il no)\n").lower()

        if string == "si":
            episodes = input("Da quanti?\n").lower()

        string = input("Sai dirmi quanto dura mediamente un episodio?").lower()

        if string == "si":
            duration = input("quanto dura mediamente?\n").lower()

    elif type_ == 'movie':

        episodes = 1

        string = input("Sai dirmi anche quanto dura il film? (si o no)\n").lower()

        if string == "si":
            duration = input("quanto dura mediamente? (esprimi la durata in minuti)\n").lower()

    string = input("Sai dirmi l'origine di questo anime? (Manga, original...)\n").lower()

    if string == "si":
        source = input("qual è?\n").lower()

    recommender = set_recommender(episodes, genre, type_, source, duration, name, score)

elif risposta == '2':
    print("ti chiedero' un po' di cose. inziamo...")
    name = input("Qual e' il nome dell'anime che vuoi classificare?\n ").lower()
    score = input("Qual e' lo score dell'anime che vuoi classificare?\n").lower()
    type_ = input("Qual e' il type dell'anime che vuoi classificare?\n").lower()
    episodes = input("Quanti episodi ha l'anime che vuoi classificare?\n").lower()
    duration = input("Quanto dura mediamente un episodio dell'anime che vuoi classificare?\n").lower()
    producers = input("Qual e' il produttore dell'anime che vuoi classificare?\n").lower()
    studios = input("Qual e' lo studio dell'anime che vuoi classificare?\n").lower()
    source = input("Qual e' l'origine dell'anime che vuoi classificare?" +
                   " (es. e' stato adattato un manga? è originale?\n").lower()

    genre_predicted = predict_genre(name, score, type_, episodes, duration, producers, studios, source)

else:
    print("sayonara")
