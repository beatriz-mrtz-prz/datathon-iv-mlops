"""Transformadores del pipeline de predicción de ventas.

Viven en un módulo, y no en el notebook, por una razón concreta: cuando MLflow
sirve el modelo con `mlflow models serve`, lo hace en un proceso aparte que tiene
que deserializar el pipeline. Si las clases estuvieran definidas en el notebook,
quedarían registradas como parte de `__main__` y ese proceso no sabría dónde
encontrarlas: el despliegue fallaría al arrancar.

Al estar aquí, basta con pasar `code_paths=["../src"]` al registrar el modelo para
que MLflow empaquete este fichero junto al modelo.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

DATE_COLUMN = "fecha_venta"
HOLIDAY_COLUMN = "festivo"
TARGET_COLUMN = "cantidad"


# --------------------------------------------------------------------------- #
# Calendario
# --------------------------------------------------------------------------- #
def add_date_columns(df: pd.DataFrame, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Descompone la fecha en día, mes, año, día de la semana y fin de semana."""
    out = df.copy()
    fechas = pd.to_datetime(out[date_column])
    out["day"] = fechas.dt.day
    out["month"] = fechas.dt.month
    out["year"] = fechas.dt.year
    out["dayofweek"] = fechas.dt.dayofweek + 1
    out["isWeekend"] = out["dayofweek"] >= 6
    return out


def add_holiday_column(df: pd.DataFrame,
                       holiday_column: str = HOLIDAY_COLUMN) -> pd.DataFrame:
    """Marca como festivo toda fila cuya columna de festivo no esté vacía."""
    out = df.copy()
    out["isHoliday"] = out[holiday_column].notna()
    return out


class CalendarTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """Convierte la fecha de venta en variables de calendario."""

    COLUMNS = ["day", "month", "year", "dayofweek", "isWeekend", "isHoliday"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = add_date_columns(X)
        out = add_holiday_column(out)
        # El índice se reinicia en los tres transformadores. El FeatureUnion los
        # concatena por columnas, y si uno conservara el índice original y otro no,
        # pandas alinearía por índice y devolvería filas duplicadas.
        return out[self.COLUMNS].reset_index(drop=True)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.COLUMNS, dtype=object)


# --------------------------------------------------------------------------- #
# Meteorología
# --------------------------------------------------------------------------- #
def get_weather_data(start: datetime, end: datetime,
                     location: tuple[float, float]) -> pd.DataFrame:
    """Descarga el parte diario de meteostat para unas coordenadas."""
    from meteostat import Daily, Point

    lat, lon = location
    return Daily(Point(lat, lon), start, end).fetch().reset_index()


def add_weather_data(df: pd.DataFrame, location: tuple[float, float],
                     date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Añade el parte meteorológico de cada fecha.

    El merge es `left` y no `inner`: con un `inner`, cualquier fecha que meteostat
    no tuviera desaparecería de los datos y el pipeline devolvería menos filas de
    las que recibió, desalineándose con el target. Con `left` se conservan todas
    las filas y los huecos los rellena después el imputador del pipeline.
    """
    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column])

    weather = get_weather_data(out[date_column].min(), out[date_column].max(), location)
    weather["time"] = pd.to_datetime(weather["time"])

    return out.merge(weather, left_on=date_column, right_on="time", how="left")


def add_rain_column(df: pd.DataFrame, precipitation_column: str = "prcp") -> pd.DataFrame:
    """Marca como lluvioso todo día con precipitación mayor que cero."""
    out = df.copy()
    out["isRaining"] = np.where(out[precipitation_column].fillna(0) > 0, True, False)
    return out


class WeatherTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """Añade temperatura media, mínima, máxima y si llovió."""

    COLUMNS = ["tavg", "tmin", "tmax", "isRaining"]

    def __init__(self, location: tuple[float, float]):
        self.location = location

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = add_weather_data(X, self.location)
        out = add_rain_column(out)
        return out[self.COLUMNS].reset_index(drop=True)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.COLUMNS, dtype=object)


# --------------------------------------------------------------------------- #
# Serie temporal a problema supervisado
# --------------------------------------------------------------------------- #
def series_to_supervised(data: pd.Series, daily_lag: int = 1,
                         weekly_lag: int = 1) -> pd.DataFrame:
    """Convierte una serie temporal en una tabla de aprendizaje supervisado.

    Cada fila pasa a llevar los valores de la variable en días y semanas
    anteriores, que es lo que el modelo usará para predecir el día presente.

    Args:
        data: la serie de observaciones.
        daily_lag: cuántos días anteriores se incorporan.
        weekly_lag: cuántas semanas anteriores se incorporan.

    Returns:
        DataFrame con una columna por retardo.
    """
    features, names = [], []

    def desplazar(lag: int, periodo: int, etiqueta: str):
        features.append(data.shift(lag * periodo))
        names.append(f"{data.name} (t-{lag} {etiqueta})")

    for lag in range(weekly_lag, 0, -1):
        desplazar(lag, 7, "weeks")
    for lag in range(daily_lag, 0, -1):
        desplazar(lag, 1, "days")

    out = pd.concat(features, axis=1)
    out.columns = names
    return out


class ToSupervisedTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """Genera los retardos diarios y semanales de la cantidad vendida."""

    def __init__(self, daily_lag: int = 6, weekly_lag: int = 4):
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        serie = X[TARGET_COLUMN].reset_index(drop=True)
        serie.name = TARGET_COLUMN
        return series_to_supervised(serie, self.daily_lag, self.weekly_lag)

    def get_feature_names_out(self, input_features=None):
        nombres = [f"{TARGET_COLUMN} (t-{lag} weeks)"
                   for lag in range(self.weekly_lag, 0, -1)]
        nombres += [f"{TARGET_COLUMN} (t-{lag} days)"
                    for lag in range(self.daily_lag, 0, -1)]
        return np.asarray(nombres, dtype=object)


# Nombres de las columnas que produce el preprocesador, en el orden del FeatureUnion
CATEGORICAL_FEATURES = CalendarTransformer.COLUMNS + ["isRaining"]
WEATHER_NUMERIC = ["tavg", "tmin", "tmax"]


def lag_feature_names(daily_lag: int = 6, weekly_lag: int = 4) -> list[str]:
    """Nombres de las columnas de retardo, para declararlas como numéricas."""
    return list(ToSupervisedTransformer(daily_lag, weekly_lag).get_feature_names_out())
