"""Conexión a la base de datos de la Panadería.

Las credenciales se leen de variables de entorno, nunca del código. El fichero
`.env` que las contiene está en `.gitignore`; `.env.example` documenta qué
variables hacen falta sin exponer ningún valor.
"""

from __future__ import annotations

import os

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()


class DatabaseConnection:
    """Cliente mínimo para leer y escribir tablas en la base de datos."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        host: str | None = None,
        port: int | None = None,
        dialect: str = "mysql",
        driver: str = "pymysql",
    ) -> None:
        username = username or os.environ["DB_USERNAME"]
        password = password or os.environ["DB_PASSWORD"]
        database = database or os.getenv("DB_NAME", "data")
        host = host or os.environ["DB_HOST"]
        port = port or int(os.getenv("DB_PORT", "3306"))

        url = sa.URL.create(
            drivername=f"{dialect}+{driver}",
            username=username,
            password=password,      # SQLAlchemy escapa los caracteres especiales
            host=host,
            port=port,
            database=database,
        )
        self.engine = sa.create_engine(url)

    def query_to_df(self, query: str) -> pd.DataFrame:
        """Ejecuta una consulta y devuelve el resultado como DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql_query(sa.text(query), conn)

    def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Escribe un DataFrame en una tabla, reemplazándola si ya existe."""
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)

    def check_connection(self) -> bool:
        """Comprueba que la conexión funciona."""
        try:
            with self.engine.connect():
                return True
        except Exception as exc:
            print(f"No se pudo conectar: {type(exc).__name__}: {exc}")
            return False
