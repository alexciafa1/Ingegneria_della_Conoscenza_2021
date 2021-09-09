'''
1) chiedere all'utente che cosa vuole fare: recommender oppure classificazione
2) prendere i dati in input e instradare il programma
3) fanculo finito
4) tu volere inculale me
'''

#def menu():
from content_based_recommender import set_recommender
from predictgenre import *

print("^^^^^^^^^^^^^^^^^^^ ANIME RECOMMENDER ^^^^^^^^^^^^^^^^^^^\n")
string = "Konichiwa, cosa vuoi fare?\n" + "Vuoi che ti suggerisca un anime ? - premi 1\n" \
         + "Vuoi sapere il genere di un anime? - premi 2\n" \
         + "Vuoi uscire? - premi 3\n"

risposta = input(string)

if risposta == '1':
    print("dimmi il genere")
    '''
    inserire il nome di un anime che ti è piaciuto
    se è un film o serie tv o ova o ona
    quanto dura?
    assegna uno score ?
    genere
    conosci la casa produttrice? si o no 
    conosci lo studio di animazione? si o no 
    sai dirmi il rating di questo anime? si o no
    conosci l'orgine di questo anime? si o no (non si fa)
    
    4 + 2^4 combinazioni =  16 = 20 casi possibili (12 casi possibili)
    genere
   '''
    name = "Yakushiji Ryouko no Kaiki Jikenbo" #1
    type_ = "TV"
    duration = "23"
    score = "7.09"#2
    source ="Novel"
    episodes ="13"
    genre ="Supernatural" #3


    recommender = set_recommender(episodes,genre, type_, source, duration, name, score)

elif risposta == '2':
    print("ti chiedero' un po' di cose. inziamo...")
    name = input(" Qual e' il nome dell'anime che vuoi classificare?\n ")
    score = input("Qual e' lo score dell'anime che vuoi classificare?\n")
    type_ = input("Qual e' il type dell'anime che vuoi classificare?\n")
    episodes = input("Quanti episodi ha l'anime che vuoi classificare?\n")
    duration = input("Quanto dura mediamente un episodio dell'anime che vuoi classificare?\n")
    producers = input("Qual e' il produttore dell'anime che vuoi classificare?\n")
    studios = input("Qual e' lo studio dell'anime che vuoi classificare?\n")
    source = input("Qual e' l'origine dell'anime che vuoi classificare? (es. e' stato adattato un manga? è originale?\n")

    # funzione predict
    genre_predicted = predict_genre(name, score, type_, episodes, duration, producers, studios, source)

    '''
    passare gli argomenti
    normalizzarli rispetto ai dizionari e al dataset
    classificare
    ottenere il numero del genere predetto
    riformattare attraverso dizionario 
    fornire risultato all'utente
    '''
else:
    print("sayonara")
