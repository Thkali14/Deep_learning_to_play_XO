import pandas as pd
import numpy as np
from random import randint
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical


def init_table():
    # Initialiser une table de jeu
    table = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    return table


def affichage_table(tab):
    # Afficher la table de jeu
    for i in range(len(tab)):
        for j in range(len(tab[i])):
            marque = ' '
            if tab[i][j] == 1:
                marque = 'X'
            elif tab[i][j] == 2:
                marque = 'O'
            if j == len(tab[i]) - 1:
                print(marque)
            else:
                print(str(marque) + "|", end='')
        if i < len(tab) - 1:
            print("-----")


def positions_disponibles(tab):
    # Fonction qui renvoie une liste des positions disponibles pour le prochain tour
    df = pd.DataFrame(tab)
    positions = []
    for col in df.columns:
        indices_lignes = df[col].loc[lambda x: x == 0].index
        positions = positions + [(line, col) for line in indices_lignes]
    return positions


def evaluation(board):
    # Fonction d'évaluation qui renvoie : 0 en cas d'égalité , 1 si X gagne, 2 si O gagne , -1 si le match est en cours

    dim = len(board)
    df = pd.DataFrame(board)
        # renvoyer le gagnant sur les diagonales
    if len(set(np.diag(df))) == 1 and df.iloc[0, 0] != 0:
        return df.iloc[0, 0]
    elif len(set(np.diag(np.fliplr(df)))) == 1 and df.iloc[0, 2]:
        return df.iloc[0, 2]
    for i in range(dim):
        # renvoyer le gagnant sur les lignes
        if len(df.iloc[i, :].unique()) == 1 and df.iloc[i, 0] != 0:
            return df.iloc[i, 0]
        # rrenvoyer le gagnant sur les colonnes
        elif len(df.iloc[:, i].unique()) == 1 and df.iloc[0, i] != 0:
            return df.iloc[i, 0]
    if not df.apply(lambda x: x.isin([0]).any()).any():
        return 0
    return -1


def simulation_jeu(joueur=None):
    # Fonction qui simule des matches aléatoires si on précise pas en entrée le joueur de départ, il est choisit au hasard
    table_jeu = init_table()
    positions = positions_disponibles(table_jeu)
    historiques = list()
    #choix aléatoire de joueur de départ
    if not joueur:
        joueur = randint(1, 2)
    while evaluation(table_jeu) == -1:
        select_position = random.choice(positions)
        table_jeu[select_position[0]][select_position[1]] = joueur
        historiques.append((joueur, select_position))
        positions.remove(select_position)
        joueur = - joueur + 3
    return historiques



def matrice_table(historiques):
    # Fonction qui convertit des données de type matrice à une table
    table = init_table()
    for elm in historiques:
        table[elm[1][0]][elm[1][1]] = elm[0]
    return table


def statistiques_jeux(jeux, joueur):
    # Afficher les statistiques des parties jouées

    gagnant = 0
    perdant = 0
    egalite = 0
    dic = {1: 'X', 2: 'O'}
    for jeu in jeux:
        resultats_table = matrice_table(jeu)
        if evaluation(resultats_table) == -1:
            continue
        elif evaluation(resultats_table) == joueur:
            gagnant += 1
        elif evaluation(resultats_table) == 0:
            egalite += 1
        else:
            perdant += 1
    pct_victoire = (gagnant / len(jeux)) * 100
    pct_perte = (perdant / len(jeux)) * 100
    pct_egalite = (egalite / len(jeux)) * 100
    print("-----Results of player %s-----\n" % (dic[joueur]))
    print("Victoire: %d (%.1f%%)\n" % (gagnant, pct_victoire))
    print("Pérte: %d (%.1f%%)\n" % (perdant, pct_perte))
    print("Egalité: %d (%.1f%%)\n" % (egalite, pct_egalite))


def modele():
    # Modèle de réseau de neurones implémentée
    outcomes = 3
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(9,)))
    model.add(Dropout(0.2))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(outcomes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


def preparer_donnees(jeux):
    #fonction qui renvoit les données d'entrainement et de test

    X = []
    y = []
    for jeu in jeux:
        gangant = evaluation(matrice_table(jeu))
        for position in range(len(jeu)):
            X.append(matrice_table(jeu[:(position + 1)]))
            y.append(gangant)

    X = np.array(X).reshape((-1, 9))
    y = to_categorical(y)
    num_entrainement = int(len(X) * 0.8)
    return (X[:num_entrainement], X[num_entrainement:], y[:num_entrainement], y[num_entrainement:])


def meilleure_decision(table, model, joueur, rnd=0):
    scores = []
    positions = positions_disponibles(table)

    #Faites des prédictions pour chaque mouvement possible
    for i in range(len(positions)):
        future = np.array(table)
        future[positions[i][0]][positions[i][1]] = joueur
        prediction = model.predict(future.reshape((-1, 9)))[0]
        if joueur == 1:
            prevision_victoire = prediction[1]
            prevision_perte = prediction[2]
        else:
            prevision_victoire = prediction[2]
            prevision_perte = prediction[1]
        prevision_egalite = prediction[0]
        if prevision_victoire - prevision_perte > 0:
            scores.append(prevision_victoire - prevision_perte)
        else:
            scores.append(prevision_egalite - prevision_perte)

    # Choisir le meilleur coup avec un facteur aléatoire
    meilleur_scores = np.flip(np.argsort(scores))
    for i in range(len(meilleur_scores)):
        if random.random() * rnd < 0.5:
            return positions[meilleur_scores[i]]

    # Choisissez un coup complètement au hasard
    return positions[random.randint(0, len(positions) - 1)]


def jouer_un_match(joueur, model):
    table = init_table()
    dic = {1: 'X', 2: 'O'}
    #cas de machine initaliser à la valeur 1 ou 2 (dans ce cas c'est la machine qui commence la partie)
    if joueur ==1 or joueur == 2:
        marque_humain = -joueur + 3
        print("**********La machine commence le jeu avec {} **********\n".format(dic[joueur]))
        while evaluation(table) == -1:
            print("******Le tour de la machine*******\n")
            machine_position = meilleure_decision(table, model, joueur)
            table[machine_position[0]][machine_position[1]] = joueur
            affichage_table(table)
            print("\n")
            print("******Votre tour:******\n")
            print("Les positions disponibles pour ce tour sont :{} \n".format(positions_disponibles(table)))
            ligne_humain_position = int(input("Veuillez choisir la position de la ligne\n"))
            colonne_humain_position = int(input("Veuillez choisir la position de la colonne\n"))
            table[ligne_humain_position][colonne_humain_position] = marque_humain
            affichage_table(table)
            print("\n")

    else:
        # ici, nous choisissons de manière aléatoire le premier (la machine ou nous) à jouer
        premier_a_jouer = randint(0, 1) # pour la valeur 0 c'est la machine qui va commencer sinon c'est nous
        if premier_a_jouer == 0:
            joueur = randint(1, 2)
            marque_humain = -joueur + 3
            print("**********La machine commence le jeu avec {} ***********\n".format(dic[joueur]))
            while evaluation(table) == -1:
                print("******Le tour de la machine*******\n")
                machine_position = meilleure_decision(table, model, joueur)
                table[machine_position[0]][machine_position[1]] = joueur
                affichage_table(table)
                print("\n")
                print("******Votre tour:******\n")
                print("Les positions disponibles pour ce tour sont :{} \n".format(positions_disponibles(table)))
                ligne_humain_position = int(input("Veuillez choisir la position de la ligne:\n"))
                colonne_humain_position = int(input("Veuillez choisir la position de la colonne:\n"))
                table[ligne_humain_position][colonne_humain_position] = marque_humain
                affichage_table(table)
                print("\n")
        else:
            print("**********On commence le jeu***********\n")
            marque_humain = int(input("Veuillez choisir votre marque: 1 pour X et 2 pour O \n"))
            joueur = -marque_humain + 3
            while evaluation(table) == -1:
                print("*******************Notre tour : les positions disponibles ***********************\n")
                print("Les positions disponibles pour ce tour sont :{} \n".format(positions_disponibles(table)))
                ligne_humain_position = int(input("Veuillez choisir la position de la ligne\n"))
                colonne_humain_position = int(input("Veuillez choisir la position de la colonne\n"))
                table[ligne_humain_position][colonne_humain_position] = marque_humain
                print("\n")
                affichage_table(table)
                print("******Le tour de la machine******\n")
                machine_position = meilleure_decision(table, model, joueur)
                table[machine_position[0]][machine_position[1]] = joueur
                affichage_table(table)
                print("\n")


def main():
    jeux = [simulation_jeu() for _ in range(1000)]
    model = modele()
    X_train, X_test, y_train, y_test = preparer_donnees(jeux)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=100)
    # Maintenant, nous jouons un jeu contre notre modèle et voyons comment il se débrouille bien
    # 1si vous voulez que la machine commence par X
    # 2 si vous voulez que la machine commence par Y
    # sinon le jeu commence au hasard (pile ou face) .
    jouer_un_match(3,model)


main()
