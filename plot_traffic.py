# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import casadi
from openap.casadi import aero as aero_casadi
from openap import aero, nav, top, prop
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, current_process
from traffic.data import navaids, airports

pd.set_option("display.max_rows", 15)
import openap
import itertools
import time
from functools import partial
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.colors as mcolors

# %%
eham = nav.airport("EHAM")
actype = "a320"
start = (eham["lat"], eham["lon"])
nodes = 39
nx, ny, nz = 50, 45, 23


rwy = "18L"
# max_fuel=False
for max_fuel in [True, False]:
    map_type = "N"
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors.extend(["b", "g", "y", "m", "c"])
    flights = pd.read_csv(f"data_generated/optimal_flights_runway{rwy}_{map_type}.csv")
    df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
    df_real = pd.read_parquet(
        f"data_generated/opensky2024_runway{rwy}_{map_type}.parquet"
    )
    c_ends = pd.read_csv(f"data_generated/opensky_runway{rwy}_ends_{map_type}.csv")
    if map_type == "DN":
        df_cost["cost"] = df_cost["cost"].apply(lambda x: x * 0.5)
    flights = flights.query("max_fuel==@max_fuel")

    cost_grid = df_cost.cost.values.reshape(nx, ny, nz)

    norm = plt.Normalize(vmin=0.00001, vmax=0.0008)
    proj = ccrs.TransverseMercator(
        central_longitude=eham["lon"], central_latitude=eham["lat"]
    )
    trans = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 6),
        subplot_kw=dict(projection=proj),
    )

    ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
    ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)
    ax.set_extent(
        [
            min(flights.longitude.values) - 0.2,
            max(flights.longitude.values) + 0.2,
            min(flights.latitude.values) - 0.2,
            max(flights.latitude.values) + 0.2,
        ]
    )

    norm = plt.Normalize(vmin=0.00001, vmax=0.03, clip=True)
    # if map_type=="DN":
    #     norm = plt.Normalize(vmin=0.00001, vmax=0.05, clip=True)
    cntr = ax.contourf(
        df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 3],
        df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 3],
        cost_grid[:, :, 3],
        cmap="binary",
        transform=trans,
        levels=20,
        norm=norm,
        alpha=0.5,
    )
    # xx=0
    # for i, fid in enumerate(flights.fid.unique()[xx:xx+1]):
    for i, fid in enumerate(flights.fid.unique()[:]):
        flightr = df_real.query(f"flight_id=='{fid}'")
        ax.plot(
            flightr.longitude,
            flightr.latitude,
            color="tab:blue",
            lw=2,
            transform=trans,
            label="Real flights centroids" if i == 0 else None,
        )
        flight = flights.query("fid==@fid and obj=='pop'")
        flight0 = flights.query("fid==@fid and obj == 'fuel'")
        ax.plot(
            flight.query("cost_grid>0").longitude,
            flight.query("cost_grid>0").latitude,
            color="k",
            lw=2,
            transform=trans,
            label="Noise-optimal" if i == 0 else None,
        )
        ax.plot(
            flight.longitude,
            flight.latitude,
            color="k",
            lw=1,
            # linestyle="dashed",
            transform=trans,
        )
        ax.plot(
            flight0.longitude,
            flight0.latitude,
            color="r",
            lw=1,
            # linestyle="dashed",
            transform=trans,
            label=f"Fuel-optimal" if i == 0 else None,
        )
    plt.title(f"{map_type}, max_fuel {max_fuel}")
    plt.legend()
    plt.tight_layout()
    if max_fuel:
        plt.savefig(f"figs/frn_{map_type}_max_fuel_{rwy}.png", bbox_inches="tight")
    else:
        plt.savefig(f"figs/frn_{map_type}_{rwy}.png", bbox_inches="tight")
    # plt.show()
# ## %%

# %%
max_fuel = False
# max_fuel=True
map_type = "D"
# for max_fuel in [True, False]:

#     for map_type in ["D","N","DN"]:
flights = pd.read_csv(
    f"data_generated/optimal_flights_runway{rwy}_{map_type}.csv"
).query("max_fuel==@max_fuel")
c_tot = 0
c0_tot = 0
f_tot = 0
f0_tot = 0

if max_fuel == True:
    coef = 0.0006
else:
    if map_type == "N":
        coef = 0.0008
    else:
        coef = 0.0009
for i, fid in enumerate(flights.fid.unique()[:]):
    flight = flights.query("fid==@fid and obj=='pop'")
    flight0 = flights.query("fid==@fid and obj=='fuel'")
    cost = (
        (coef * flight.cost_grid * flight.thrust) * flight.ts.diff().values[-1]
    ).sum()
    cost0 = (
        (coef * flight0.cost_grid * flight0.thrust) * flight0.ts.diff().values[-1]
    ).sum()
    fuel = (flight.mass.values[0] - flight.mass.values[-1]) / (
        flight0.mass.values[0] - flight0.mass.values[-1]
    )
    cost_grid = flight.cost_grid.sum() / flight0.cost_grid.sum()
    print(
        i,
        fid[:5],
        "\tcost",
        f"{cost / cost0:.4f}",
        "\tfuel",
        f"{fuel:.4f}",
        "\tcost_grid",
        f"{cost_grid:.4f}",
    )
    f_tot = f_tot + (flight.mass.values[0] - flight.mass.values[-1])
    f0_tot = f0_tot + (flight0.mass.values[0] - flight0.mass.values[-1])
    c_tot = c_tot + cost
    c0_tot = c0_tot + cost0
    # if i==36:
    #     break
print(
    "\t\tcost_t",
    f"{c_tot / c0_tot:.4f}",
    "\tfuel_t",
    f"{f_tot / f0_tot:.4f}",
    "\tcost_grid_t",
    round(
        flights.query("obj=='pop'").cost_grid.sum()
        / flights.query("obj=='fuel'").cost_grid.sum(),
        4,
    ),
)


# %%

# %%
max_fuel = False
map_type = "N"
# for max_fuel in [True, False]:

#     for map_type in ["D","N","DN"]:
flights = pd.read_csv(
    f"data_generated/optimal_flights_runway{rwy}_{map_type}.csv"
).query("max_fuel==@max_fuel")
c_tot = 0
c0_tot = 0
f_tot = 0
f0_tot = 0
for i, fid in enumerate(flights.fid.unique()[:]):
    flight = flights.query("fid==@fid and obj=='pop'")
    thrust = openap.Thrust("a320")
    drag = openap.Drag("a320", wave_drag=True)
    D = drag.clean(
        mass=flight.mass, tas=flight.tas, alt=flight.altitude, vs=flight.vertical_rate
    )
    gamma = np.arctan2(
        flight.vertical_rate * openap.aero.fpm, flight.tas * openap.aero.kts
    )
    flight = flight.assign(thrust2=D + flight.mass * 9.81 * np.sin(gamma))

    flight0 = flights.query("fid==@fid and obj=='fuel'")
    thrust = openap.Thrust("a320")
    drag = openap.Drag("a320")
    D = drag.clean(
        mass=flight0.mass,
        tas=flight0.tas,
        alt=flight0.altitude,
        vs=flight0.vertical_rate,
    )
    gamma = np.arctan2(
        flight0.vertical_rate * openap.aero.fpm, flight0.tas * openap.aero.kts
    )
    flight0 = flight0.assign(thrust2=D + flight0.mass * 9.81 * np.sin(gamma))
    cost = ((flight.cost_grid * flight.thrust2) * flight.ts.diff().values[-1]).sum()
    cost0 = ((flight0.cost_grid * flight0.thrust2) * flight0.ts.diff().values[-1]).sum()
    fuel = (flight.mass.values[0] - flight.mass.values[-1]) / (
        flight0.mass.values[0] - flight0.mass.values[-1]
    )
    cost_grid = flight.cost_grid.sum() / flight0.cost_grid.sum()
    print(
        i,
        fid[:5],
        "\tcost",
        f"{cost / cost0:.4f}",
        "\tfuel",
        f"{fuel:.4f}",
        "\tcost_grid",
        f"{cost_grid:.4f}",
    )
    f_tot = f_tot + (flight.mass.values[0] - flight.mass.values[-1])
    f0_tot = f0_tot + (flight0.mass.values[0] - flight0.mass.values[-1])
    c_tot = c_tot + cost
    c0_tot = c0_tot + cost0
    # print(abs(flight.thrust2-flight.thrust).sum())
    # if i==36:
    #     break
print(
    "\t\tcost_t",
    f"{c_tot / c0_tot:.4f}",
    "\tfuel_t",
    f"{f_tot / f0_tot:.4f}",
    "\tcost_grid_t",
    round(
        flights.query("obj=='pop'").cost_grid.sum()
        / flights.query("obj=='fuel'").cost_grid.sum(),
        4,
    ),
)


# %%
