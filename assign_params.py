# %%
from traffic.core import Traffic, Flight
import pandas as pd
import numpy as np
from openap import FuelFlow, Thrust, aero, top, Drag
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer, CRS
import itertools
from tqdm import tqdm
import sys
import os
import casadi
import seaborn as sns

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath("../src/"))

from pathlib import Path

root_dir = Path(__file__).resolve().parent


# %%
def assign_ff_etc(flight, interpolant, m0, obj):
    ac = "a320"
    # map_type = flight.map_type.iloc[0]
    # if obj=="pop_night":
    #     cdn=0.0009
    # else:
    #     cdn=0.0001
    cdn = 0.0009
    drag = Drag(ac, wave_drag=True)
    thrust = Thrust(ac)
    fuelflow = FuelFlow(ac)
    mass_ = np.array([m0] * len(flight))
    for i in range(5):
        ff = fuelflow.enroute(
            mass=mass_, tas=flight.tas, alt=flight.altitude, vs=flight.vertical_rate
        )
        mass_ = (
            np.array([m0] * len(flight)) - (ff * flight.ts.diff().fillna(0)).cumsum()
        )

    flight = flight.assign(
        mass=mass_,
        fuelflow=ff,
    )
    cost = interpolant(
        np.array(
            [
                flight.longitude.values,
                flight.latitude.values,
                flight.altitude.values * aero.ft,
            ]
        )
    )
    D = drag.clean(
        mass=flight.mass,
        tas=flight.tas,
        alt=flight.altitude,
        vs=flight.vertical_rate,
    )
    gamma = np.arctan2(flight.vertical_rate * aero.fpm, flight.tas * aero.kts)
    T = D + flight.mass * 9.81 * np.sin(gamma)
    T_idle = thrust.descent_idle(flight.tas, flight.altitude)
    T_max = thrust.climb(tas=flight.tas, alt=flight.altitude, roc=flight.vertical_rate)
    # T[T < thrust_idle] = thrust_idle[T < thrust_idle]

    T = (
        (
            np.log(1 + np.exp(20 * (T - T_idle * 0.8) / 100_000))
            - np.log(1 + np.exp(20 * (T - T_max * 1.2) / 100_000))
        )
        / (np.log(1 + np.exp(20)))
    ) * 100_000 + T_idle * 0.8

    flight = flight.assign(
        thrust=T,
        fuel=lambda x: x.fuelflow * x.ts.diff().fillna(0),
        # fuel=lambda x: x.mass.max()-x.mass.min(),
    )
    flight = flight.assign(grid_cost=cost.full()[0])
    flight = flight.assign(
        obj_T=lambda x: cdn * x.grid_cost * x.thrust.abs() * x.ts.diff().bfill()
    )
    flight = flight.assign(obj_w=lambda x: x.obj_T + x.fuel)
    # flight.loc[flight.query("altitude>18_000").index, "obj_T"] = 0
    # flight.loc[flight.query("altitude>18_000").index, "grid_cost"] = 0

    return flight


# %%

cols = [
    "ts",
    "latitude",
    "longitude",
    "altitude",
    "h",
    "vertical_rate",
    "mass",
    "tas",
    "fid",
    # "frnd",
    # "map_type",
    "obj",
]
interp = "linear"
interp = "bspline"
if interp == "bspline":
    interpolant = top.tools.load_interpolant(
        path=f"{root_dir}/data_generated/cashed_interp_DN_15k.casadi"
    )
else:
    df_cost = pd.read_parquet(f"{root_dir}/data_generated/df_cost_DN_ext.parquet")
    interpolant = top.tools.interpolant_from_dataframe(df_cost)
l = []
opts = pd.read_csv(f"{root_dir}/data_generated/optimal_flights_DN_39.csv")

flights = []
for fid in opts.fid.unique()[:]:
    m0 = opts.query("fid==@fid and obj=='pop_night'").mass.max()
    for obj in opts.obj.unique()[:]:
        f = opts.query("fid==@fid and obj==@obj")[cols]
        f_cost = assign_ff_etc(f, interpolant=interpolant, m0=m0, obj=obj)
        flights.append(f_cost)
        l.append(
            {
                "fid": fid,
                "obj": f_cost.obj.iloc[0],
                "fuel": f_cost.fuel.sum(),
                "grid_cost": f_cost.grid_cost.sum(),
                "obj_T": f_cost.obj_T.sum(),
                "obj_w": f_cost.obj_w.sum(),
            }
        )
flights = pd.concat(flights)
flights.to_parquet(f"{root_dir}/data_generated/flight_params_frnd_DN.parquet")
df = pd.DataFrame(l)
df.to_csv(f"{root_dir}/data_generated/compare_params_frnd_DN.parquet")

df = df.sort_values(by="fid")

sns.barplot(data=df, hue="obj", y="obj_T")

# %%
sns.barplot(data=df, hue="obj", y="fuel")


# %%

df = pd.read_parquet(f"{root_dir}/../data/trajs/{op}_flights_frn.parquet")
flights = df.query("obj=='pop'")
flights0 = df.query("obj=='fuel'")
f_tot = 0
f0_tot = 0
for i, fid in enumerate(df.fid.unique()[:]):

    flight = flights.query("fid==@fid")
    flight0 = flights0.query("fid==@fid")
    # cost = ((0.02*flight.cost_grid*flight.thrust+flight.fuelflow)*flight.ts.diff().values[-1]).sum()
    # cost0 = ((0.02*flight0.cost_grid*flight0.thrust+flight0.fuelflow)*flight0.ts.diff().values[-1]).sum()
    cost = ((flight.grid_cost * flight.thrust) * flight.ts.diff().fillna(0)).sum()
    cost0 = ((flight0.grid_cost * flight0.thrust) * flight0.ts.diff().fillna(0)).sum()
    # cost = flight.obj_T.sum()
    # cost0 = flight0.obj_T.sum()
    fuel = (flight.mass.values[0] - flight.mass.values[-1]) / (
        flight0.mass.values[0] - flight0.mass.values[-1]
    )
    grid_cost = flight.grid_cost.sum() / flight0.grid_cost.sum()

    cost = cost * 0.00025 + flight.mass.values[0] - flight.mass.values[-1]
    cost0 = cost0 * 0.00025 + flight0.mass.values[0] - flight0.mass.values[-1]
    # cost=cost*0.00025+(flight.fuelflow * flight.ts.diff().fillna(0)).sum()
    # cost0=cost0*0.00025+(flight0.fuelflow * flight0.ts.diff().fillna(0)).sum()

    print(
        i,
        fid[:5],
        "\tcost",
        f"{cost/cost0:.4f}",
        "\tfuel",
        f"{fuel:.4f}",
        "\tcost_grid",
        f"{grid_cost:.4f}",
    )
    f_tot = f_tot + (flight.mass.values[0] - flight.mass.values[-1])
    f0_tot = f0_tot + (flight0.mass.values[0] - flight0.mass.values[-1])
print(
    "cost_t",
    round(flights.grid_cost.sum() / flights0.grid_cost.sum(), 4),
    "\tfuel_t",
    f_tot / f0_tot,
)
# %%
import matplotlib.pyplot as plt

opt0 = pd.read_parquet(f"{root_dir}/../data/trajs/{op}_flights_frn_old.parquet")
opt1 = pd.read_parquet(f"{root_dir}/../data/trajs/{op}_flights_frn.parquet")
for i, fid in enumerate(df.fid.unique()[:]):
    flight1 = opt1.query("obj=='pop' and fid==@fid")
    flightf = opt1.query("obj=='fuel' and fid==@fid")
    flight0 = opt0.query("obj=='pop' and fid==@fid")
    flightr = opt0.query("obj=='real' and fid==@fid")
    plt.plot(flight1.ts, flight1.altitude, label="new")
    plt.plot(flight0.ts, flight0.altitude, label="old")
    plt.plot(flightr.ts, flightr.altitude, label="real")
    plt.plot(flightf.ts, flightf.altitude, label="fuel")
    print("old", flight0.fuel.sum())
    print("new", flight1.fuel.sum())
    print("real", flightr.fuel.sum())
    print("fuel", flightf.fuel.sum())
    plt.suptitle(fid)
    plt.legend()
    plt.show()
# %%
