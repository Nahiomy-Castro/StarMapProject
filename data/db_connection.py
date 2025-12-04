import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
import sqlalchemy as sqal
import mariadb

def create_MariaDB():
    DB_CONFIG = {
        "host": "localhost",
        "user": "coen2220",
        "password": "coen2220",
        "port": 3306
    }

    DB_ADMIN = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '<PASSWORD>',
    }

    NEW_DB_NAME = 'group01'

    conn = None
    cursor = None
    try:
        conn = mariadb.connect(**DB_ADMIN)
        cursor = conn.cursor()

        print("Connected to MariaDB")

        try:
            create_db_query = f"CREATE DATABASE IF NOT EXISTS {NEW_DB_NAME}"
            cursor.execute(create_db_query)
            print(f"Database '{NEW_DB_NAME}' successfully created.")
        except mariadb.Error as e:
            print(f"Error creating database: {e}")
            conn.rollback()

        try:
            grant_privileges_query = f"GRANT ALL PRIVILEGES ON {NEW_DB_NAME}.* TO '{DB_CONFIG['user']}'@'{DB_CONFIG['host']}'"
            cursor.execute(grant_privileges_query)
            print("Privileges granted successfully.")
        except mariadb.Error as e:
            print(f"Error granting privileges: {e}")
            conn.rollback()
        try:
            cursor.execute("FLUSH PRIVILEGES")
            print("Privileges flushed.")
            conn.commit()
        except mariadb.Error as e:
            print(f"Error flushing privileges: {e}")
            conn.rollback()

    except mariadb.Error as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()
        print("Database connection closed.")



def upload_MariaDB():
    DB_CONFIG = {
        "host": "localhost",
        "user": "coen2220",
        "password": "coen2220",
        "port": 3306,
        "database": "group01"
    }

    engine = sqal.create_engine(
        f"mariadb+mariadbconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )


    # hyg = Table.read("hyg_v42.csv", format="csv", encoding="utf-8")
    # df_hyg = hyg.to_pandas()
    # df_hyg.to_sql("hyg_4_2", engine, if_exists="append", index=False, method="multi", chunksize=500)
    #
    # print("HYG 4.2 Database uploaded successfully.\n")

    # cons_88 = Table.read("88-constellations.csv", format="csv", encoding="utf-8")
    # df_cons88 = cons_88.to_pandas()
    # df_cons88.to_sql("cons_88", engine, if_exists="append", index=False, method="multi", chunksize=500)

    # cons = Table.read("constellations.csv", format="csv", encoding="utf-8")
    # df_cons = cons.to_pandas()
    # df_cons.to_sql("cons", engine, if_exists="append", index=False, method="multi", chunksize=500)
    #
    # print("Constellations Database uploaded successfully.\n")
    #
    # synthstar = Table.read("star_dataset.csv", format="csv", encoding="utf-8")
    # df_synth = synthstar.to_pandas()
    # df_synth.to_sql("synthset", engine, if_exists="append", index=False, method="multi", chunksize=500)
    #
    # print("Synthetic Database uploaded successfully.\n")
    #
    # hygtest = Table.read("hyg_v42_updated_test.csv", format="csv", encoding="utf-8")
    # df_hygtest = hygtest.to_pandas()
    # df_hygtest.to_sql("hyg_4_2_updated_test", engine, if_exists="append", index=False, method="multi", chunksize=500)
    #
    # print("HYG 4.2 Test Database uploaded successfully.\n")
    #
    #
    hygupdated = Table.read("hyg_v42_updated.csv", format="csv", encoding="utf-8")
    df_hygupdated = hygupdated.to_pandas()
    df_hygupdated.to_sql("hyg_4_2_updated", engine, if_exists="append", index=False, method="multi", chunksize=500)

    print("HYG 4.2 Updated Database uploaded successfully.\n")

    print("All databases uploaded successfully.\n")


if __name__ == "__main__":

    #create_MariaDB()
    upload_MariaDB()



