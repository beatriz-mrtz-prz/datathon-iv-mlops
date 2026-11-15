"""Reconstruye el dataset de estudio a partir de los Excel de origen.

Replica en pandas la misma lógica que las vistas SQL del notebook de carga:

1. `articulos_top`    — ordena los artículos por importe dentro de cada familia.
2. `calendario`       — cruza las fechas con el fichero de festivos.
3. `ventas_diarias`   — agrega las ventas horarias a nivel de día.
4. `ventas_diarias_estudio` — filtra ventas (no mermas), dentro del periodo de
   estudio, de los 5 artículos top de cada familia y anteriores a mayo de 2023.

Existe para que el proyecto se pueda reproducir sin acceso a la base de datos.
Uso:

    python src/prepare_data.py --raw raw --out data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

INICIO_ESTUDIO = pd.Timestamp("2021-05-01")
FIN_ESTUDIO = pd.Timestamp("2023-05-01")
TOP_N = 5


def cargar_excels(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ventas = pd.read_excel(raw / "ArticulosPanaderia.xlsx")
    calendario = pd.read_excel(raw / "Calendario.xlsx")
    return ventas, calendario


def articulos_top(ventas: pd.DataFrame) -> pd.DataFrame:
    """Ordena los artículos por importe acumulado dentro de cada familia."""
    periodo = ventas[ventas["FechaVenta"] >= INICIO_ESTUDIO]
    totales = (periodo.groupby(["FAMILIA", "Articulo"], as_index=False)["Importe"]
               .sum()
               .rename(columns={"Importe": "importe_total"}))
    totales["orden_articulo_familia"] = (
        totales.groupby("FAMILIA")["importe_total"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return totales.drop(columns="importe_total")


def ventas_diarias(ventas: pd.DataFrame, calendario: pd.DataFrame,
                   orden: pd.DataFrame) -> pd.DataFrame:
    """Agrega las ventas horarias a nivel diario y añade festivo y orden."""
    df = ventas.copy()
    df["importe_ponderado"] = df["Precio"] * df["Cantidad"]

    diario = (df.groupby(["FAMILIA", "Tipo", "FechaVenta", "Articulo"], as_index=False)
              .agg(cantidad=("Cantidad", "sum"),
                   importe=("Importe", "sum"),
                   importe_ponderado=("importe_ponderado", "sum")))

    # El precio del día es el precio medio ponderado por unidades: a lo largo de
    # la jornada puede haber cambios de precio sobre el mismo artículo.
    diario["precio"] = diario["importe_ponderado"] / diario["cantidad"]
    diario = diario.drop(columns="importe_ponderado")

    festivos = (calendario.drop_duplicates(subset="Fecha")
                .rename(columns={"Fecha": "FechaVenta", "Festivo": "festivo"}))
    diario = diario.merge(festivos, on="FechaVenta", how="left")
    diario = diario.merge(orden, on=["FAMILIA", "Articulo"], how="inner")

    return diario.rename(columns={"FAMILIA": "familia", "Tipo": "tipo",
                                  "FechaVenta": "fecha_venta",
                                  "Articulo": "articulo"})


def filtrar_estudio(diario: pd.DataFrame) -> pd.DataFrame:
    """Aplica los filtros de la vista de estudio."""
    mask = (
        (diario["tipo"] == "VENTA")
        & (diario["fecha_venta"] >= INICIO_ESTUDIO)
        & (diario["fecha_venta"] < FIN_ESTUDIO)
        & (diario["orden_articulo_familia"] <= TOP_N)
    )
    columnas = ["familia", "articulo", "fecha_venta", "festivo", "precio",
                "cantidad", "importe", "orden_articulo_familia"]
    return (diario.loc[mask, columnas]
            .sort_values(["articulo", "fecha_venta"])
            .reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("raw"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--product-id", type=int, default=3960)
    parser.add_argument("--horizon", type=str, default="2023-04-19",
                        help="Primera fecha del conjunto de test.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ventas, calendario = cargar_excels(args.raw)
    estudio = filtrar_estudio(ventas_diarias(ventas, calendario, articulos_top(ventas)))
    estudio.to_csv(args.out / "ventas_diarias_estudio.csv", index=False)
    print(f"Vista de estudio: {estudio.shape[0]} filas, "
          f"{estudio['articulo'].nunique()} artículos")

    item = estudio[estudio["articulo"] == args.product_id].reset_index(drop=True)
    if item.empty:
        raise SystemExit(f"El artículo {args.product_id} no está en la vista de estudio.")

    horizonte = pd.Timestamp(args.horizon)
    train = item[item["fecha_venta"] < horizonte]
    test = item[item["fecha_venta"] >= horizonte]

    train.to_csv(args.out / "train.csv", index=False)
    test.to_csv(args.out / "test.csv", index=False)
    print(f"Artículo {args.product_id}: {len(train)} filas de train "
          f"({train['fecha_venta'].min().date()} a {train['fecha_venta'].max().date()}), "
          f"{len(test)} de test "
          f"({test['fecha_venta'].min().date()} a {test['fecha_venta'].max().date()})")


if __name__ == "__main__":
    main()
