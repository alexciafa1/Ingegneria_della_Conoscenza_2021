import os

import prediction_genre
import recommender


def main():
    bol = False
    while True:
        print("^^^^^^^^^^^^^^^^^^^ ANIME RECOMMENDER ^^^^^^^^^^^^^^^^^^^\n")
        if bol:
            string = "Vuoi che ti suggerisca un altro anime ? - Premi 1\n" \
                     + "Vuoi sapere il genere di un altro anime? - Premi 2\n" \
                     + "Vuoi uscire? - Premi 3\n"

        else:
            string = "Konichiwa, cosa vuoi fare?\n" + "Vuoi che ti suggerisca un anime ? - Premi 1\n" \
                     + "Vuoi sapere il genere di un anime? - Premi 2\n" \
                     + "Vuoi uscire? - Premi 3\n"
        response = input(string)

        if response == '1':
            # RECOMMENDER ******************************************************************

            name = ''
            score = ''
            genre = ''
            type_ = ''
            duration = ''
            episodes = ''
            source = ''

            print("Ti faro' delle domande per poterti suggerire un anime.")

            name = input("Suggeriscimi il nome di un anime che hai apprezzato\n").lower()

            score = input("Se dovessi valutarlo che voto gli daresti da 1 a 10?\n").lower()

            genre = input("Qual è il genere di questo anime?\n").lower()

            string = input("Sai dirmi il tipo di questo anime? (Si o No)(se non sai cos'è il tipo digita 1)\n").lower()

            while string == '1':

                print("Quando parliamo di tipo intendiamo se l'anime appartiene a una di queste categorie:\n" +
                      "- TV      : l'anime viene trasmesso ad episodi\n" +
                      "- Movie   : l'anime è un film\n" +
                      "- ONA/OVA : l'anime è una puntata speciale di un anime o originale\n")

                string = input("Ora sai dirmi qual è il tipo? (Si o No)(se non sai cos'è il tipo digita 1)\n").lower()

            if string == "si":
                type_ = input("Qual è il tipo?\n").lower()

            if type_ == 'tv' or type_.lower() == 'ona' or type_.lower() == 'ova':

                string = input("Sai dirmi da quanti episodi è formato? (Si o No)\n").lower()

                if string == "si":
                    episodes = input("Da quanti?\n").lower()

                string = input("Sai dirmi quanto dura mediamente un episodio?\n").lower()

                if string == "si":
                    duration = input("Quanto dura mediamente?\n").lower()

            elif type_ == 'movie':

                episodes = 1

                string = input("Sai dirmi anche quanto dura il film? (si o no)\n").lower()

                if string == "si":
                    duration = input("quanto dura mediamente? (esprimi la durata in minuti)\n").lower()

            string = input("Sai dirmi l'origine di questo anime? (Si o No)(se non sai cos'è l'origine digita 1)\n").lower()
            while string == '1':

                print("Quando parliamo di origine intendiamo se l'anime appartiene a una di queste categorie:\n" +
                      "- Manga\n- Novel\n- Book \n- Radio\n- Picture book\n- Web Manga\n- Digital Manga\n- Original\n")

                string = input("Ora sai dirmi qual è l'origine? (Si o No)(se non sai cos'è l'origine digita 1)\n").lower()

            if string == "si":

                source = input("Qual è?\n").lower()

            recommender.main(episodes, genre, type_, source, duration, name, score)
            print("\n")
            os.system("pause")
            bol = True
            print("\n")


        elif response == '2':

            print("Ti chiedero' un po' di cose. Inziamo...")

            name = input("Qual è il nome dell'anime che vuoi classificare?\n ").lower()

            score = input("Qual è lo score dell'anime che vuoi classificare?\n").lower()

            type_ = input("Qual è il type dell'anime che vuoi classificare?(se non sai cos'è il tipo digita 1)\n").lower()

            while type_ == '1':

                print("Quando parliamo di tipo intendiamo se l'anime appartiene a una di queste categorie:\n" +
                      "- TV      : l'anime viene trasmesso ad episodi\n" +
                      "- Movie   : l'anime è un film\n" +
                      "- ONA/OVA : l'anime è una puntata speciale di un anime o originale\n")

                type_ = input(
                    "Quindi, qual è il type dell'anime che vuoi classificare?(se non sai cos'è il tipo digita 1)\n").lower()

            episodes = input("Quanti episodi ha l'anime che vuoi classificare?\n").lower()

            duration = input("Quanto dura mediamente un episodio dell'anime che vuoi classificare?\n").lower()

            producers = input("Qual è il produttore dell'anime che vuoi classificare?\n").lower()

            studios = input("Qual è lo studio dell'anime che vuoi classificare?\n").lower()

            source = input("Qual è l'origine dell'anime che vuoi classificare?" +
                           " (se non sai cos'è l'origine di un anime tipo digita 1)\n").lower()

            while source == '1':

                print("Quando parliamo di origine intendiamo se l'anime appartiene a una di queste categorie:\n" +
                      "- Manga\n- Novel\n- Book \n- Radio\n- Picture book\n- Web Manga\n- Digital Manga\n- Original\n")

                source = input("Quindi, qual è l'origine dell'anime che vuoi classificare?(se non sai cos'è l'origine "
                               "digita 1)\n").lower()

            prediction_genre.predict_genre(name, score, type_, episodes, duration, producers, studios, source)
            print("\n")
            os.system("pause")
            bol = True
            print("\n\n")

        else:
            print("Sayonara")
            break


if __name__ == '__main__':
    main()
