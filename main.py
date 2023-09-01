import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from decision_tree import decision_tree

from knn import knn


if __name__ == "__main__":
    engine = create_engine('postgresql://postgres:bonjour@localhost:5432/big_data')
    db_connection = engine.connect()

    # Get datas
    sql_query = text('SELECT "Vie_economique".code_departement, "Vie_economique".date_donnees, taux_chomage, nb_entreprise, "Vote".taux, "Candidat".indicateur_partis, "Securite".nb_effraction from "Vie_economique" join "Vote" on "Vote".code_departement = "Vie_economique".code_departement and "Vote".date_donnees = "Vie_economique".date_donnees join "Candidat" on "Candidat".id = "Vote".id_candidat join "Securite" on "Vie_economique".code_departement = "Securite".code_departement and "Vie_economique".date_donnees = "Securite".date_donnees')
    df = pd.read_sql_query(sql=sql_query, con=db_connection)

    # Clean datas
    df["taux_with_indicateur_partis"] = (df["taux"] * df["indicateur_partis"])/100
    df.drop(['taux', 'indicateur_partis'], axis=1, inplace=True)
    df = df.groupby(['code_departement', 'date_donnees']).agg({
        'taux_with_indicateur_partis': 'sum',
        'taux_chomage': 'first',
        'nb_entreprise': 'first',
        'nb_effraction': 'first'
    }).reset_index()

    print(df.corr(numeric_only=True)['taux_with_indicateur_partis'])

    X = df[['taux_chomage', 'nb_entreprise', 'nb_effraction']]
    y = df['taux_with_indicateur_partis']

    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=12345)

    knn(X_train, y_train, X_test, y_test)

    decision_tree(X_train, y_train, X_test, y_test)