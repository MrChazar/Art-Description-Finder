import sqlalchemy as sal
import pyodbc
import pandas as pd


SERVER_NAME = 'localhost'
DATABASE_NAME = 'projekt_database'

engine = sal.create_engine(f'mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?'
                           'driver=ODBC Driver 17 for SQL Server&Trusted_Connection=yes')


def save_list_to_database(data, table_name):
    flat_list = [item for sublist in data for item in sublist]
    df = pd.DataFrame(flat_list)
    df.to_sql(table_name, con=engine, index=False, if_exists='append')


def read_database_to_pandas(query, database=engine):
    return pd.read_sql(query, database)