import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text


if __name__ == "__main__":
    engine = create_engine('postgresql://postgres:bonjour@localhost:5432/big_data')
    db_connection = engine.connect()

    # Get datas
    sql_query = text('SELECT "Vie_economique".code_departement, "Vie_economique".date_donnees, taux_chomage, nb_entreprise, "Vote".taux, "Candidat".indicateur_partis  from "Vie_economique" join "Vote" on "Vote".code_departement = "Vie_economique".code_departement and "Vote".date_donnees = "Vie_economique".date_donnees join "Candidat" on "Candidat".id = "Vote".id_candidat')
    df = pd.read_sql_query(sql=sql_query, con=db_connection)

    # Clean datas
    df["taux_with_indicateur_partis"] = (df["taux"] * df["indicateur_partis"])/100
    df.drop(['taux', 'indicateur_partis'], axis=1, inplace=True)
    df = df.groupby(['code_departement', 'date_donnees']).agg({
        'taux_with_indicateur_partis': 'sum',
        'taux_chomage': 'first',
        'nb_entreprise': 'first'
    }).reset_index()

    print(df)