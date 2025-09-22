#%%
from traffic.core import Traffic,Flight
import pandas as pd
import numpy as np
from openap import FuelFlow, Thrust, aero, top
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer, CRS
import itertools
from tqdm import tqdm
import sys
import os

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath("../src/"))
from pp_affected import pp_affected, cost_grid_cost, get_npd_interpolator
#%%
ac= "a320"
fuelflow = FuelFlow(ac)
thrust = Thrust(ac)

curr_path = os.path.dirname(os.path.realpath(__file__))
options = itertools.product(["N","D","DN"],[50,60,75], ["f","r","n"])

for map_type,treshold,frn in options:
    print(map_type,treshold,frn)
    if map_type=="D" or map_type=="N":
        pop_map = pd.read_parquet(curr_path+ f"/../data_raw/{map_type}032011_1K_cropped.parquet").rename(columns = {"popul":"pp"}).fillna(0)
    else:
        pop_map = pd.read_parquet(curr_path+ f"/../data_raw/pop_static.parquet")
        
    flights = pd.read_csv(f"../data_generated/optimal_flights_{map_type}.csv")
    if frn=="f":
        flights = flights.query("obj=='pop'")
    elif frn=="n":   
        flights = flights.query("obj=='pop'")
    else:
        flights = (pd.read_parquet(curr_path+ f"/../data_generated/opensky2024_centroids_{map_type}.parquet")
        .assign(frn = "r")
        .rename(columns={"groundspeed":"tas", "flight_id":"fid"})[[
        'ts',
        'fid',
        'latitude', 
        'altitude',
        'longitude',
        'mass',
        'vertical_rate',
        'tas',
        'fuel',
        'thrust',
        'h',
        'frn', ]])
    df_cost = pd.read_csv(curr_path+ f"/../data_generated/df_cost_{map_type}.csv")

    pp_dict =[]

    for fid in flights.fid.unique():
        flight = flights.query("fid==@fid").reset_index(drop=True)
        if map_type=="DN":
            grid_type = "ll"
        else:
            grid_type ="xy"
        flight = cost_grid_cost(df_cost, flight)
        pp_aff_all, pp_aff_uni, flight = pp_affected(ac, pop_map, grid_type, flight, treshold)
        pp_dict.append(
            {
                "fid": fid,
                "fuel": flight.fuel.sum(),
                "cost_grid": flight.cost_grid.sum(),
                "people_affected_sum_all": pp_aff_all,
                "pp_unique_sum": pp_aff_uni,
                "map_type":map_type,
                "frn": frn,
            }
        )

    df = pd.DataFrame.from_dict(pp_dict)
    df.to_csv(
        curr_path+ f"/../data_generated/pp_affected_{treshold}/pp_affected_frn_{map_type}.csv",
        index=False,
    )

# %%
