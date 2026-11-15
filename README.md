# Predicción de ventas — Panadería Salvador

Predicción de la demanda diaria de un producto de panadería, con el ciclo
completo de MLOps: pipeline de scikit-learn, seguimiento de experimentos en
MLflow, registro del modelo, despliegue como API REST y escritura de las
predicciones en la base de datos.

Datathon IV — Máster en Data Analytics y Data Science, Code Space Academy.

---

## El problema

Panadería Salvador produce cada día sin saber cuánto va a vender. Pasarse genera
merma: producto que se tira. Quedarse corto es venta perdida y un cliente que se
va a otro sitio. El objetivo es estimar las unidades que se venderán de un
artículo en una fecha concreta, para ajustar la producción.

Se trabaja con el artículo **3960** (familia BOLLERÍA), el de mayor facturación
de su familia. El periodo de estudio va de mayo de 2021 a abril de 2023.

Es un **problema de regresión sobre una serie temporal**: la variable objetivo es
`cantidad`, las unidades vendidas en un día.

---

## Enfoque

### Las variables

El modelo combina tres bloques de información, cada uno con su propio
transformador:

| Bloque | Transformador | Variables que aporta |
|---|---|---|
| Calendario | `CalendarTransformer` | día, mes, año, día de la semana, fin de semana, festivo |
| Meteorología | `WeatherTransformer` | temperatura media, mínima y máxima, y si llovió |
| Histórico | `ToSupervisedTransformer` | ventas de los 6 días y las 4 semanas anteriores |

Los datos meteorológicos se obtienen de la API de [meteostat](https://dev.meteostat.net/)
para las coordenadas de la panadería.

El bloque histórico es el que convierte la serie temporal en un problema
supervisado: cada fila incorpora lo que se vendió en días y semanas anteriores.
Los retardos semanales importan porque la demanda de panadería tiene un ciclo
semanal muy marcado: un sábado se parece mucho más al sábado anterior que al
viernes.

### El pipeline

```
TransformedTargetRegressor
└── Pipeline
    ├── FeatureUnion ......... calendario + meteorología + retardos
    ├── ColumnTransformer .... OneHotEncoder (categóricas) + MinMaxScaler (numéricas)
    ├── SimpleImputer ........ rellena los retardos de los primeros días
    └── SVR .................. el modelo
```

Todo el preprocesamiento vive **dentro** del pipeline. Esto no es cosmético: hace
que el escalado y la codificación se ajusten solo con los datos de entrenamiento
de cada partición, y que el objeto que se despliega sepa transformar los datos
crudos por sí mismo. La API REST recibe las columnas tal y como salen de la base
de datos.

El `TransformedTargetRegressor` escala también la variable objetivo, porque el
SVR es sensible a la escala del target.

### Validación

La validación cruzada usa **`TimeSeriesSplit`**, no `KFold`. En una serie
temporal con variables de retardo, barajar las particiones entrena el modelo con
días posteriores a los que luego evalúa: el modelo ve el futuro y la métrica sale
inflada. `TimeSeriesSplit` respeta el orden cronológico, que es como el modelo
funcionará en producción.

La partición final de test son los últimos días de la serie, a partir del
**19 de abril de 2023**, que nunca se usan para entrenar.

Las métricas registradas son **R²**, **MAE** y **RMSE**. El R² dice qué parte de
la variabilidad explica el modelo; el MAE y el RMSE están en unidades de producto,
que es lo que el panadero entiende: cuántas piezas se desvía la predicción.

---

## Estructura del repositorio

```
.
├── notebooks/
│   └── datathon_iv_mlops.ipynb   Desarrollo completo, documentado paso a paso
├── src/
│   ├── transformers.py           Transformadores del pipeline
│   ├── database.py               Conexión a la base de datos
│   └── prepare_data.py           Reconstruye el dataset desde los Excel de origen
├── data/                         Datos (ignorado por Git)
├── requirements.txt
├── .env.example                  Plantilla de variables de entorno
└── .gitignore
```

Los transformadores están en `src/` y no en el notebook **a propósito**. Cuando
MLflow sirve el modelo, lo hace en un proceso aparte que debe deserializar el
pipeline. Si las clases estuvieran definidas en el notebook quedarían asociadas a
`__main__` y ese proceso no sabría encontrarlas: el despliegue fallaría al
arrancar. Al vivir en un módulo, basta con pasar `code_paths` al registrar el
modelo para que MLflow lo empaquete junto a él.

---

## Puesta en marcha

### 1. Entorno

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Credenciales

```bash
cp .env.example .env
```

Rellena `.env` con los datos de conexión. **Ese fichero no se sube nunca**: está
en `.gitignore` y es la razón por la que no hay ninguna contraseña en el código.

### 3. Datos

Si tienes acceso a la base de datos, el notebook los descarga solo. Si no, se
reconstruyen desde los Excel de origen colocándolos en `raw/`:

```bash
python src/prepare_data.py --raw raw --out data
```

### 4. Servidor de MLflow

En una terminal aparte:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

La interfaz queda en <http://localhost:5000>.

### 5. Ejecutar el notebook

```bash
jupyter lab notebooks/datathon_iv_mlops.ipynb
```

### 6. Desplegar el modelo

Cuando el notebook haya registrado el modelo, en otra terminal:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "models:/datathon_iv@production" -p 5001 --env-manager local
```

Las últimas celdas del notebook hacen peticiones a esa API y escriben las
predicciones en la base de datos.

---

## Seguridad

La plantilla original del ejercicio traía el usuario y la contraseña de la base
de datos escritos directamente en una celda. Este repositorio no los contiene:

- Las credenciales se leen de variables de entorno con `python-dotenv`.
- `.env` está en `.gitignore`.
- `.env.example` documenta qué variables hacen falta, sin ningún valor.
- `data/`, `raw/`, `mlruns/`, `mlartifacts/` y `mlflow.db` también están ignorados:
  ni los datos del cliente ni los artefactos generados pertenecen al repositorio.

---

## Flujo de trabajo con ramas

El desarrollo se hizo en ramas, integradas en `main` mediante Pull Request:

| Rama | Contenido |
|---|---|
| `feature/data-pipeline` | Transformadores, conexión a la base de datos y reconstrucción del dataset |
| `feature/mlflow-tracking` | Notebook: pipeline, seguimiento, registro y despliegue del modelo |
| `feature/docs` | Documentación del proyecto |

---

## Resultados

Las métricas de la última ejecución quedan registradas en MLflow, en el
experimento **Datathon IV**, junto con los parámetros, los gráficos y el propio
modelo. La sección final del notebook las recoge y comenta.

---

## Licencia y datos

Los datos pertenecen a Panadería Salvador y se usan con fines formativos dentro
del máster. No se incluyen en este repositorio.
