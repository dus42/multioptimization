#%%
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

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
#%%
@click.command()
@click.option("--map_type", required=True, help="DN, N or D")
@click.option("--plot", is_flag=True, default=False)

def main(map_type, plot):
    # map = "DN"
    nx = 30
    ny = 30
    nz = 20
    eham = nav.airport("EHAM")
    ends = pd.read_csv(f"../data_generated/opensky_centroid_ends_{map_type}.csv")[
        ["latitude", "longitude"]
    ].values


    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
    transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)

    start = (eham["lat"], eham["lon"])

    min_lon,min_lat = (min(start[1], min(ends[:, 1])) - 0.3, min(start[0], min(ends[:, 0])) - 0.4)
    max_lon,max_lat = (max(start[1], max(ends[:, 1])) + 0.3, max(start[0], max(ends[:, 0])) + 0.4)
    # min_lola = (4, 51.5)
    # max_lola = (6, 53)
    min_xy = transformer_xy.transform(min_lon, min_lat)
    max_xy = transformer_xy.transform(max_lon, max_lat)
    bounds = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
    if map_type == "D" or map_type == "N":
        pop_map = pd.read_parquet(f"../data_raw/{map_type}012011_1K_cropped.parquet").rename(columns = {"popul":"pp"})
        grid_type = "xy" 

    elif map_type == "DN":
        pop_map = pd.read_parquet(f"../data_raw/pop_static.parquet")
        grid_type = "ll" 

    df_cost = cost_generator(pop_map, grid_type, 
                            alt_max = 20000,  
                            nodes = (nx,ny,nz), 
                            airport = "EHAM",
                            bounds = bounds, 
                            max_res_len=400,
                            plot = plot)
    df_cost.to_csv(f"../data_generated/df_cost_{map_type}.csv", index=False)

# %%
if __name__ == "__main__":
    main()
# %%
