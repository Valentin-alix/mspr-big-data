import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from decision_tree import decision_tree

from knn import knn
from random_forest import random_forest


if __name__ == "__main__":
    engine = create_engine('postgresql://postgres:bonjour@localhost:5432/big_data')
    db_connection = engine.connect()

    # Get datas
    sql_query = text('SELECT "Vie_economique".code_departement,"Vie_economique".salaire_net_horaire_moyen, "Vie_economique".date_donnees, taux_chomage, nb_entreprise, "Vote".taux, "Candidat".indicateur_partis, "Securite".nb_effraction, "Demographie".taux_genre from "Vie_economique" join "Vote" on "Vote".code_departement = "Vie_economique".code_departement and "Vote".date_donnees = "Vie_economique".date_donnees join "Candidat" on "Candidat".id = "Vote".id_candidat join "Securite" on "Vie_economique".code_departement = "Securite".code_departement and "Vie_economique".date_donnees = "Securite".date_donnees join "Demographie" on "Demographie".code_departement = "Securite".code_departement and "Demographie".date_donnees = "Securite".date_donnees')
    df = pd.read_sql_query(sql=sql_query, con=db_connection)

    # Clean datas
    df["taux_with_indicateur_partis"] = (df["taux"] * df["indicateur_partis"])/100
    df.drop(['taux', 'indicateur_partis'], axis=1, inplace=True)
    df = df.groupby(['code_departement', 'date_donnees']).agg({
        'taux_with_indicateur_partis': 'sum',
        'taux_chomage': 'first',
        'nb_entreprise': 'first',
        'nb_effraction': 'first',
        'taux_genre': 'first',
        'salaire_net_horaire_moyen': 'first'
    }).reset_index()

    print(df.corr(numeric_only=True)['taux_with_indicateur_partis'])

    X_train = df[(df['date_donnees'] > datetime(2007,1,1).date()) & (df['date_donnees'] <= datetime(2017,1,1).date())][['taux_chomage', 'nb_entreprise', 'nb_effraction', 'taux_genre', 'salaire_net_horaire_moyen']]
    X_test = df[(df['date_donnees'] > datetime(2017,1,1).date()) & (df['date_donnees'] <= datetime(2022,1,1).date())][['taux_chomage', 'nb_entreprise', 'nb_effraction', 'taux_genre', 'salaire_net_horaire_moyen']]

    y_train =  df[(df['date_donnees'] == datetime(2012,1,1).date()) | (df['date_donnees'] == datetime(2017,1,1).date())][['taux_with_indicateur_partis']]
    y_test = df[(df['date_donnees'] == datetime(2022,1,1).date())][['taux_with_indicateur_partis']]

    knn(X_train, y_train, X_test, y_test)

    decision_tree(X_train, y_train, X_test, y_test)

    random_forest(X_train, y_train, X_test, y_test)