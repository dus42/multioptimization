# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import casadi
from openap.casadi import aero as aero_casadi
from openap import aero, nav, prop
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, current_process
from traffic.data import navaids, airports

pd.set_option("display.max_rows", 15)

import openap
from openap import top
import itertools

import time
from functools import partial
import warnings
from pathlib import Path

root_dir = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")

# %%
flights = []
for map_type in ["D", "N", "DN"]:
    centrs0 = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
    for fid in tqdm(centrs0.fid.values[:]):
        centrs = centrs0.query("fid==@fid")
        rwy = airports["EHAM"].runways.data.query(f"name=='{centrs.runway.values[0]}'")
        start = (rwy.latitude.values[0], rwy.longitude.values[0])
        actype = "a320"
        end = (centrs.latitude.iloc[0], centrs.longitude.iloc[0])
        # trk_ = centrs.runway_dir.iloc[0]
        m0 = centrs.tow.iloc[0] / 78000
        alt = centrs.altitude.iloc[0]

        nodes = 10

        # generate fuel optimal trajectory
        done = False
        while done == False:
            optimizer = top.Climb(actype, start, end, m0=m0)
            optimizer.setup(nodes=nodes, debug=False)
            flight = optimizer.trajectory(obj="fuel", h_end=alt * aero.ft)

            if flight is not None:
                if abs(flight.altitude.max() - alt) < 300:
                    done = True
            if m0 > 0.6:
                m0 = m0 * 0.98
            else:
                print(f"{fid} does not optimize")
                break
        if done != False:
            flights.append(flight.assign(fid=fid))

    df = pd.concat(flights)
    df.to_csv(
        f"cruise_to_find_mass_{map_type}.csv",
        index=False,
    )
# %%
df_real = pd.read_parquet(
    f"{root_dir}/../scripts/{op}s_centrs_resampled.parquet"
).query("flight_id==@fid")
# %%
l = []
for map_type in ["D", "N", "DN"][2:]:
    df = pd.read_csv(f"cruise_to_find_mass_{map_type}.csv")
    for fid in df.fid.unique():
        l.append(
            {
                "fid": fid,
                "map_type": map_type,
                "m0_new": df.query("fid==@fid").mass.max(),
                "alt_new": df.query("fid==@fid").altitude.max(),
            }
        )
    centrs0 = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
    centrs0.merge(pd.DataFrame(l), on="fid")
    # .to_csv(
    #     f"{root_dir}/../data/trajs/centroids/centroid_start_end_{map_type}.csv",
    #     index=False,
    # )
# %%
