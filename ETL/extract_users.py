
import pandas as pd
from pandas import DataFrame

from sqlalchemy import create_engine, MetaData, Table, text, Engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session


SQLALCHEMY_DATABASE_URL = "postgresql://postgres:admin@localhost:5432/picta"

engine = create_engine(
  SQLALCHEMY_DATABASE_URL, 
)

print('Conectando con la base de datos...')
Session = sessionmaker(bind=engine)
session = Session()


def extract_users():
    sql_consult = """
    SELECT id, username, first_name, last_name, phone_number, email, fecha_nacimiento 
    from app_usuario;
"""
    amount = 100_000

    print('Extrayendo los datos...')
    # Crear un motor que se conecta a la base de datos
    with engine.connect() as connection:
        result = list(connection.execute(text(sql_consult)))

    # Convertir el resultado a un DataFrame de pandas
    df = pd.DataFrame(result, columns=['id', 'username', 'first_name', 'last_name', 'phone_number', 'email', 'fecha_nacimiento'])

    # Limitar el DataFrame a la cantidad de filas especificada por 'amount'
    # df_limited = df.head(amount)

    # Guardar el DataFrame limitado en un archivo CSV
    df.to_csv('usuarios.csv', index=False)

    return df

if __name__ == "__main__":
  extract_users()