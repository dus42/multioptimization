# %%
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from pyproj import Transformer, CRS
from openap import aero, nav
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import click
import sys
import os

sys.path.append(os.path.abspath("../src/"))
from cost_generation import cost_generator


# %%
@click.command()
@click.option("--map_type", required=True, help="DN, N or D")
@click.option("--plot", is_flag=True, default=False)
# @click.option("--month", default="03")
# def main(map_type, month, plot):
def main(map_type, plot):
    # map = "DN"
    nx = 100
    ny = 100
    nz = 27
    eham = nav.airport("EHAM")

    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
    transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)

    start = (eham["lat"], eham["lon"])

    # min_lon,min_lat = (min(start[1], min(ends[:, 1])) - 0.3, min(start[0], min(ends[:, 0])) - 0.4)
    # max_lon,max_lat = (max(start[1], max(ends[:, 1])) + 0.3, max(start[0], max(ends[:, 0])) + 0.4)
    min_lon, min_lat = (1.7, 49)
    max_lon, max_lat = (7.5, 54)
    min_xy = transformer_xy.transform(min_lon, min_lat)
    max_xy = transformer_xy.transform(max_lon, max_lat)
    bounds = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]

    # pop_map = pd.read_parquet(f"../data_raw/grid_3035_xy.parquet").rename(
    #     columns={"value": "pp"}
    # )
    # grid_type = "xy"
    # pop_x = pop_map.x.unique()
    # pop_y = pop_map.y.unique()
    # lon, lat = transformer_ll.transform(pop_map.x.values, pop_map.y.values)
    # pop_map = pop_map.assign(lon=lon, lat=lat)
    if map_type == "DN":
        pop_map = pd.read_parquet(f"../data_raw/landscan2023.parquet").rename(
            columns={"value": "pp", "x": "lon", "y": "lat"}
        )
        grid_type = "ll"
    elif map_type == "N":
        pop_map = pd.read_parquet(f"../data_raw/N032011_1K_cropped.parquet").rename(
            columns={"value": "pp"}
        )
        grid_type = "xy"
    else:
        pop_map = pd.read_parquet(f"../data_raw/D032011_1K_cropped.parquet").rename(
            columns={"value": "pp"}
        )
        grid_type = "xy"
    df_cost = cost_generator(
        pop_map,
        grid_type,
        alt_max=25000,
        nodes=(nx, ny, nz),
        airport="EHAM",
        bounds=bounds,
        plot=plot,
    )
    df_cost.to_parquet(f"../data_generated/df_cost_{map_type}_new.parquet", index=False)


# %%
if __name__ == "__main__":
    main()
# %%
