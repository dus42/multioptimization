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
import warnings

warnings.filterwarnings("ignore")


# %%
def obj_pop_exposure_day(x, u, dt, optimizer=None, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = np.arctan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.00015 * cost * thrust * dt + fuel


def obj_pop_exposure_night(x, u, dt, optimizer=None, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = np.arctan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.0004 * cost * thrust * dt + fuel


def obj_max_fuel(x, u, dt, optimizer=None, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = np.arctan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.001 * cost * thrust * dt + fuel


# %%


def opt_trajs(fid, map_type, is_max_fuel=False):
    flight_real = pd.read_parquet(
        "data_generated/opensky2024_centroids_DN.parquet"
    ).query("flight_id==@fid")
    eham = nav.airport("EHAM")
    actype = "a320"
    start = (eham["lat"], eham["lon"])
    nodes = 39

    if is_max_fuel:
        obj = obj_max_fuel
    else:
        if map_type == "N":
            obj = obj_pop_exposure_night
        else:
            obj = obj_pop_exposure_day

    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv").query(
        "fid==@fid"
    )
    month = c_ends.month.values[0]
    month = f"{month:02d}"
    if map_type == "DN":
        month = ""
    df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}{month}.csv")
    if map_type != "DN":
        df_cost["cost"] = df_cost["cost"].apply(lambda x: x * 2)

    rwy = airports["EHAM"].runways.data.query(f"name=='{c_ends.runway.values[0]}'")
    if rwy is None or len(rwy) == 0:
        trk_start = None
        start = "EHAM"
    else:
        trk_start = rwy.bearing.values[0]
        start = (rwy.latitude.values[0], rwy.longitude.values[0])

    m0 = c_ends.tow.values[0] / prop.aircraft(actype)["mtow"]
    if m0 > 0.95:
        m0 = 0.99 * m0
    end = (c_ends.latitude.values[0], c_ends.longitude.values[0])
    h_end = c_ends.altitude.values[0] * aero.ft

    # generate fuel optimal trajectory
    optimizer = top.Climb(actype, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=10000, debug=False)
    flight_fuel = optimizer.trajectory(
        objective="fuel",
        h_end=h_end,
        runway_dir=trk_start,
    )
    if flight_fuel is None:
        m0 = 0.999 * m0
        optimizer = top.Climb(actype, start, end, m0=m0)
        optimizer.setup(nodes=nodes, max_iteration=10000, debug=False)
        flight_fuel = optimizer.trajectory(
            objective="fuel",
            h_end=h_end,
            runway_dir=trk_start,
        )
    if flight_fuel is None:
        print(fid, "fuel optimization failed")
        return
    flight_fuel = flight_fuel.assign(fid=fid, obj="fuel")
    drag = openap.Drag("a320", wave_drag=True)
    D = drag.clean(
        mass=flight_fuel.mass,
        tas=flight_fuel.tas,
        alt=flight_fuel.altitude,
        vs=flight_fuel.vertical_rate,
    )
    gamma = np.arctan2(
        flight_fuel.vertical_rate * openap.aero.fpm, flight_fuel.tas * openap.aero.kts
    )
    flight_fuel = flight_fuel.assign(thrust=D + flight_fuel.mass * 9.81 * np.sin(gamma))

    flight_real = flight_real.assign(fid=fid, obj="real")
    drag = openap.Drag("a320", wave_drag=True)
    D = drag.clean(
        mass=flight_real.mass,
        tas=flight_real.groundspeed,
        alt=flight_real.altitude,
        vs=flight_real.vertical_rate,
    )
    gamma = np.arctan2(
        flight_real.vertical_rate * openap.aero.fpm,
        flight_real.groundspeed * openap.aero.kts,
    )
    flight_real = flight_real.assign(thrust=D + flight_real.mass * 9.81 * np.sin(gamma))
    flight_real = flight_real.assign(
        mach=lambda x: openap.aero.tas2mach(x.groundspeed * openap.aero.kts, x.h)
    ).rename(columns={"groundspeed": "tas"})

    # generate optimalpopulation exposure trajectory
    if map_type == "N" and is_max_fuel:
        max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.02
    elif not is_max_fuel:
        max_fuel = None
    else:
        max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.005

    optimizer = top.Climb(actype, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=35000, debug=False)

    objective_with_opt = partial(obj, optimizer=optimizer)

    interpolant = top.tools.interpolant_from_dataframe(df_cost)

    flight_pop = optimizer.trajectory(
        objective=objective_with_opt,
        interpolant=interpolant,
        h_end=h_end,
        runway_dir=trk_start,
        max_fuel=max_fuel,
        initial_guess=flight_fuel,
    )
    if flight_pop is None:
        m0 = 0.99 * m0
        optimizer = top.Climb(actype, start, end, m0=m0)
        optimizer.setup(nodes=nodes, max_iteration=35000, debug=False)

        objective_with_opt = partial(obj, optimizer=optimizer)

        interpolant = top.tools.interpolant_from_dataframe(df_cost)

        flight_pop = optimizer.trajectory(
            objective=objective_with_opt,
            interpolant=interpolant,
            h_end=h_end,
            runway_dir=trk_start,
            max_fuel=max_fuel,
            initial_guess=flight_fuel,
        )

    if flight_pop is None:
        print(fid, "noise optimization failed")
        return
    flight_pop = flight_pop.assign(fid=fid, obj="pop")

    drag = openap.Drag("a320", wave_drag=True)
    D = drag.clean(
        mass=flight_pop.mass,
        tas=flight_pop.tas,
        alt=flight_pop.altitude,
        vs=flight_pop.vertical_rate,
    )
    gamma = np.arctan2(
        flight_pop.vertical_rate * openap.aero.fpm, flight_pop.tas * openap.aero.kts
    )
    flight_pop = flight_pop.assign(thrust=D + flight_pop.mass * 9.81 * np.sin(gamma))
    # calculate cost from grid cost
    cost = interpolant(
        np.array(
            [
                flight_pop.longitude.values,
                flight_pop.latitude.values,
                flight_pop.h.values,
            ]
        )
    )
    cost0 = interpolant(
        np.array(
            [
                flight_fuel.longitude.values,
                flight_fuel.latitude.values,
                flight_fuel.h.values,
            ]
        )
    )
    costr = interpolant(
        np.array(
            [
                flight_real.longitude.values,
                flight_real.latitude.values,
                flight_real.h.values,
            ]
        )
    )
    flight_pop = flight_pop.assign(cost_grid=cost.full()[0], max_fuel=is_max_fuel)
    flight_fuel = flight_fuel.assign(cost_grid=cost0.full()[0], max_fuel=is_max_fuel)
    flight_real = flight_real.assign(cost_grid=costr.full()[0], max_fuel=is_max_fuel)[
        [
            "mass",
            "ts",
            "x",
            "y",
            "h",
            "latitude",
            "longitude",
            "altitude",
            "mach",
            "tas",
            "vertical_rate",
            "heading",
            "fuelflow",
            "fid",
            "obj",
            "thrust",
            "cost_grid",
            "max_fuel",
        ]
    ]

    return pd.concat([flight_fuel, flight_pop, flight_real])


# %%
actype = "a320"
start = airports["EHAM"].latlon

for map_type in ["DN"]:
    # for map_type in ["D", "N", "DN"]:

    # for map_type in ["DN"]:
    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

    # options = itertools.product(c_ends.fid.values, [map_type], [True, False])
    options = itertools.product(c_ends.fid.values, [map_type], [False])
    options = (
        pd.DataFrame(options, columns=["fid", "map_type", "is_max_fuel"])
        .sort_values(by=["is_max_fuel", "map_type"])
        .reset_index(drop=True)
    )

    flights = []

    with ProcessPoolExecutor(max_workers=6) as executor:
        tasks = {
            executor.submit(opt_trajs, fid, map_type, is_max_fuel)
            for i, (fid, map_type, is_max_fuel) in options.iterrows()
        }
        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            ncols=0,
            desc=f"generating: {map_type}",
        ):
            fr = future.result()
            flights.append(fr)

    pd.concat(flights, ignore_index=False).to_csv(
        f"data_generated/optimal_flights_{map_type}.csv", index=False
    )


# %%
eham = nav.airport("EHAM")
actype = "a320"
start = (eham["lat"], eham["lon"])
nodes = 39


import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.colors as mcolors


for max_fuel in [False, True]:
    map_type = "N"
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors.extend(["b", "g", "y", "m", "c"])
    flights = pd.read_csv(f"data_generated/optimal_flights_{map_type}.csv")
    df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
    df_real = pd.read_parquet(
        f"data_generated/opensky2024_centroids_{map_type}.parquet"
    )

    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

    flights = flights.query("max_fuel==@max_fuel")
    nx, ny, nz = 50, 45, 23
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
    if map_type == "DN":
        norm = plt.Normalize(vmin=0.00001, vmax=0.05, clip=True)
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
            label="Population-optimal" if i == 0 else None,
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
            linestyle="dashed",
            transform=trans,
            label=f"Fuel-optimal" if i == 0 else None,
        )
    # plt.title(f"{map_type}, max_fuel {max_fuel}")
    plt.legend()
    plt.tight_layout()
    if max_fuel:
        plt.savefig(f"figs/frn_{map_type}_max_fuel.png", bbox_inches="tight")
    else:
        plt.savefig(f"figs/frn_{map_type}.png", bbox_inches="tight", dpi=300)
    plt.show()
## %%
# %%
map_type = "N"
max_fuel = True
max_fuel = False
flights = pd.read_csv(f"data_generated/optimal_flights_{map_type}.csv").query(
    "max_fuel==@max_fuel"
)
f_tot = 0
f0_tot = 0
print(map_type)
for i, fid in enumerate(flights.fid.unique()[:]):
    flight = flights.query("fid==@fid and obj=='pop'")
    flight0 = flights.query("fid==@fid and obj=='fuel'")
    cost = ((flight.cost_grid * flight.thrust) * flight.ts.diff().values[-1]).sum()
    cost0 = ((flight0.cost_grid * flight0.thrust) * flight0.ts.diff().values[-1]).sum()
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
print(
    "cost_t",
    round(
        flights.query("obj=='pop'").cost_grid.sum()
        / flights.query("obj=='fuel'").cost_grid.sum(),
        4,
    ),
    "\tfuel_t",
    f_tot / f0_tot,
)

# %%
map_type = "DN"
max_fuel = True
max_fuel = False
flights = pd.read_csv(f"data_generated/optimal_flights_{map_type}.csv").query(
    "max_fuel==@max_fuel"
)
c_ends = c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
flights = flights.merge(c_ends[["cluster", "fid"]], on="fid")
f_tot = 0
f0_tot = 0
print(map_type)
l = []
for i, fid in enumerate(flights.fid.unique()[:]):
    flight = flights.query("fid==@fid and obj=='pop'")
    flight0 = flights.query("fid==@fid and obj=='fuel'")
    flightr = flights.query("fid==@fid and obj=='real'")
    cost = ((flight.cost_grid * flight.thrust) * flight.ts.diff().values[-1]).sum()
    cost0 = ((flight0.cost_grid * flight0.thrust) * flight0.ts.diff().values[-1]).sum()
    costr = ((flightr.cost_grid * flightr.thrust) * flightr.ts.diff().values[-1]).sum()
    fuel = flight.mass.values[0] - flight.mass.values[-1]
    fuel0 = flight0.mass.values[0] - flight0.mass.values[-1]
    fuelr = flightr.mass.max() - flightr.mass.min()
    # cost_grid = flight.cost_grid.sum() / flight0.cost_grid.sum()
    l.append(
        {
            "cluster": flight.cluster.iloc[0],
            "cost_p/cost_f": cost / cost0,
            "fuel_p/fuel_f": fuel / fuel0,
        }
    )

df = (
    pd.DataFrame()
    .from_dict(l)
    .sort_values(by="cluster")
    .set_index("cluster", drop=True)
)
df["cost_p/cost_f"] = df["cost_p/cost_f"].apply(lambda x: f"{x:.04f}")
df["fuel_p/fuel_f"] = df["fuel_p/fuel_f"].apply(lambda x: f"{x:.04f}")
print(df.to_latex())


# %%
# %%
# max_fuel = True
max_fuel = False

print(map_type)
l = []
for dn, map_type in zip(["Day", "Night"], ["D", "N"]):
    flights = pd.read_csv(f"data_generated/optimal_flights_{map_type}.csv").query(
        "max_fuel==@max_fuel"
    )
    c_ends = c_ends = pd.read_csv(
        f"data_generated/opensky_centroid_ends_{map_type}.csv"
    )
    flights = flights.merge(c_ends[["cluster", "fid"]], on="fid")
    f_tot = 0
    f0_tot = 0
    for i, fid in enumerate(flights.fid.unique()[:]):
        flight = flights.query("fid==@fid and obj=='pop'")
        flight0 = flights.query("fid==@fid and obj=='fuel'")
        flightr = flights.query("fid==@fid and obj=='real'")
        cost = ((flight.cost_grid * flight.thrust) * flight.ts.diff().values[-1]).sum()
        cost0 = (
            (flight0.cost_grid * flight0.thrust) * flight0.ts.diff().values[-1]
        ).sum()
        costr = (
            (flightr.cost_grid * flightr.thrust) * flightr.ts.diff().values[-1]
        ).sum()
        fuel = flight.mass.values[0] - flight.mass.values[-1]
        fuel0 = flight0.mass.values[0] - flight0.mass.values[-1]
        fuelr = flightr.mass.max() - flightr.mass.min()
        # cost_grid = flight.cost_grid.sum() / flight0.cost_grid.sum()
        l.append(
            {
                "cluster": flight.cluster.iloc[0],
                "dn": dn,
                "cost_p/cost_f": cost / cost0,
                "fuel_p/fuel_f": fuel / fuel0,
            }
        )

df = (
    pd.DataFrame()
    .from_dict(l)
    .sort_values(by="cluster")
    .set_index(["cluster", "dn"], drop=True)
)

df = pd.DataFrame().from_dict(l).drop(columns="cluster").groupby("dn").mean()
df["cost_p/cost_f"] = df["cost_p/cost_f"].apply(lambda x: f"{x:.04f}")
df["fuel_p/fuel_f"] = df["fuel_p/fuel_f"].apply(lambda x: f"{x:.04f}")
print(df.to_latex())
# %%


# def opt_trajs200(fid, map_type, is_max_fuel=False, mach_end=0.8, nodes=100):
#     eham = nav.airport("EHAM")
#     actype = "a320"
#     start = (eham["lat"], eham["lon"])

#     if is_max_fuel:
#         obj = obj_max_fuel
#     else:
#         if map_type == "N":
#             obj = obj_pop_exposure_night
#         else:
#             obj = obj_pop_exposure_day

#     c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_DN.csv").query(
#         "fid==@fid"
#     )
#     month = 3
#     month = f"{month:02d}"
#     if map_type == "DN":
#         month = ""
#     df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}{month}.csv")
#     if map_type != "DN":
#         df_cost["cost"] = df_cost["cost"].apply(lambda x: x * 2)

#     rwy = airports["EHAM"].runways.data.query(f"name=='{c_ends.runway.values[0]}'")
#     if rwy is None or len(rwy) == 0:
#         trk_start = None
#         start = "EHAM"
#     else:
#         trk_start = rwy.bearing.values[0]
#         start = (rwy.latitude.values[0], rwy.longitude.values[0])

#     m0 = c_ends.tow.values[0] / prop.aircraft(actype)["mtow"]
#     if m0 > 0.95:
#         m0 = 0.99 * m0
#     end = (c_ends.latitude.values[0], c_ends.longitude.values[0])
#     h_end = c_ends.altitude.values[0] * aero.ft

#     # generate fuel optimal trajectory
#     optimizer = top.Climb(actype, start, end, m0=m0)
#     optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)
#     flight_fuel = optimizer.trajectory(
#         objective="fuel",
#         h_end=h_end,
#         runway_dir=trk_start,
#         # mach_end=mach_end,
#     )
#     if flight_fuel is None:
#         m0 = 0.999 * m0
#         optimizer = top.Climb(actype, start, end, m0=m0)
#         optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)
#         flight_fuel = optimizer.trajectory(
#             objective="fuel",
#             h_end=h_end,
#             runway_dir=trk_start,
#             # mach_end=mach_end,
#         )
#     if flight_fuel is None:
#         print(fid, "fuel optimization failed")
#         return
#     flight_fuel = flight_fuel.assign(fid=fid, obj="fuel", map_type=map_type)
#     drag = openap.Drag("a320", wave_drag=True)
#     D = drag.clean(
#         mass=flight_fuel.mass,
#         tas=flight_fuel.tas,
#         alt=flight_fuel.altitude,
#         vs=flight_fuel.vertical_rate,
#     )
#     gamma = np.arctan2(
#         flight_fuel.vertical_rate * openap.aero.fpm, flight_fuel.tas * openap.aero.kts
#     )
#     flight_fuel = flight_fuel.assign(thrust=D + flight_fuel.mass * 9.81 * np.sin(gamma))

#     # generate optimalpopulation exposure trajectory
#     if map_type == "N" and is_max_fuel:
#         max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.02
#     elif not is_max_fuel:
#         max_fuel = None
#     else:
#         max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.01

#     optimizer = top.Climb(actype, start, end, m0=m0)
#     optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)

#     objective_with_opt = partial(obj, optimizer=optimizer)

#     interpolant = top.tools.interpolant_from_dataframe(df_cost)

#     flight_pop = optimizer.trajectory(
#         objective=objective_with_opt,
#         interpolant=interpolant,
#         h_end=h_end,
#         runway_dir=trk_start,
#         max_fuel=max_fuel,
#         initial_guess=flight_fuel,
#         # mach_end=mach_end,
#     )

#     if flight_pop is None:
#         print(fid, "noise optimization failed")
#         return
#     flight_pop = flight_pop.assign(fid=fid, obj="pop")

#     drag = openap.Drag("a320", wave_drag=True)
#     D = drag.clean(
#         mass=flight_pop.mass,
#         tas=flight_pop.tas,
#         alt=flight_pop.altitude,
#         vs=flight_pop.vertical_rate,
#     )
#     gamma = np.arctan2(
#         flight_pop.vertical_rate * openap.aero.fpm, flight_pop.tas * openap.aero.kts
#     )
#     flight_pop = flight_pop.assign(thrust=D + flight_pop.mass * 9.81 * np.sin(gamma))
#     # calculate cost from grid cost
#     cost = interpolant(
#         np.array(
#             [
#                 flight_pop.longitude.values,
#                 flight_pop.latitude.values,
#                 flight_pop.h.values,
#             ]
#         )
#     )
#     cost0 = interpolant(
#         np.array(
#             [
#                 flight_fuel.longitude.values,
#                 flight_fuel.latitude.values,
#                 flight_fuel.h.values,
#             ]
#         )
#     )
#     flight_pop = flight_pop.assign(
#         cost_grid=cost.full()[0], max_fuel=is_max_fuel, map_type=map_type
#     )
#     flight_fuel = flight_fuel.assign(
#         cost_grid=cost0.full()[0], max_fuel=is_max_fuel, map_type=map_type
#     )

#     return pd.concat([flight_fuel, flight_pop])


# # %%
# ### optimize 1 fligth
# fid = "VLG12BR_219"
# df_real = (
#     pd.read_parquet(f"data_generated/opensky2024_centroids_DN.parquet")
#     .query("flight_id=='VLG12BR_219'")
#     .reset_index(drop=True)
# ).assign(mach=lambda x: aero.tas2mach(x.groundspeed * aero.kts, x.h))
# end = pd.read_csv(f"data_generated/opensky_centroid_ends_DN.csv").query("fid==@fid")
# mach_end = df_real.mach.values[-1]
# mach_end = None
# actype = "a320"
# start = airports["EHAM"].latlon

# # options = itertools.product([fid], ["D", "N"], [True, False], [mach_end])
# options = itertools.product([fid], ["D", "N"], [False], [mach_end])
# options = (
#     pd.DataFrame(options, columns=["fid", "map_type", "is_max_fuel", "mach_end"])
#     .sort_values(by=["is_max_fuel", "map_type"])
#     .reset_index(drop=True)
# )

# flights = []

# with ProcessPoolExecutor(max_workers=4) as executor:
#     tasks = {
#         executor.submit(opt_trajs200, fid, map_type, is_max_fuel, mach_end, 150)
#         for i, (fid, map_type, is_max_fuel, mach_end) in options.iterrows()
#     }
#     for future in tqdm(
#         as_completed(tasks),
#         total=len(tasks),
#         ncols=0,
#         desc=f"generating: {fid}",
#     ):
#         fr = future.result()
#         flights.append(fr)

# pd.concat(flights, ignore_index=False).to_csv(
#     f"data_generated/200_optimal_flights_new.csv", index=False
# )


# # %%

# eham = nav.airport("EHAM")
# actype = "a320"
# start = (eham["lat"], eham["lon"])


# import cartopy.crs as ccrs
# from cartopy.feature import BORDERS, COASTLINE
# import matplotlib.colors as mcolors


# max_fuel = False
# for max_fuel in [False]:
#     map_type = "D"
#     colors = list(mcolors.TABLEAU_COLORS.keys())
#     colors.extend(["b", "g", "y", "m", "c"])
#     flights = pd.read_csv(f"data_generated/200_optimal_flights.csv").query(
#         "map_type==@map_type"
#     )
#     df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
#     df_real = pd.read_parquet(f"data_generated/opensky2024_centroids_DN.parquet")
#     c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_DN.csv")

#     flights = flights.query("max_fuel==@max_fuel")
#     nx, ny, nz = 50, 45, 23
#     cost_grid = df_cost.cost.values.reshape(nx, ny, nz)

#     norm = plt.Normalize(vmin=0.00001, vmax=0.0008)
#     proj = ccrs.TransverseMercator(
#         central_longitude=eham["lon"], central_latitude=eham["lat"]
#     )
#     trans = ccrs.PlateCarree()
#     fig, ax = plt.subplots(
#         1,
#         1,
#         figsize=(6, 6),
#         subplot_kw=dict(projection=proj),
#     )

#     ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
#     ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)
#     ax.set_extent(
#         [
#             min(flights.longitude.values) - 0.2,
#             max(flights.longitude.values) + 0.2,
#             min(flights.latitude.values) - 0.2,
#             max(flights.latitude.values) + 0.2,
#         ]
#     )

#     norm = plt.Normalize(vmin=0.00001, vmax=0.03, clip=True)
#     if map_type == "DN":
#         norm = plt.Normalize(vmin=0.00001, vmax=0.05, clip=True)
#     cntr = ax.contourf(
#         df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 3],
#         df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 3],
#         cost_grid[:, :, 3],
#         cmap="binary",
#         transform=trans,
#         levels=20,
#         norm=norm,
#         alpha=0.5,
#     )
#     # xx=0
#     # for i, fid in enumerate(flights.fid.unique()[xx:xx+1]):
#     for i, fid in enumerate(flights.fid.unique()[:]):
#         flightr = df_real.query(f"flight_id=='{fid}'")
#         ax.scatter(
#             flightr.longitude,
#             flightr.latitude,
#             color="tab:blue",
#             s=2,
#             transform=trans,
#             label="Real flights centroids" if i == 0 else None,
#         )
#         flight = flights.query("fid==@fid and obj=='pop'")
#         flight0 = flights.query("fid==@fid and obj == 'fuel'")
#         # ax.scatter(
#         #     flight.query("cost_grid>0").longitude,
#         #     flight.query("cost_grid>0").latitude,
#         #     color="k",
#         #     s=2,
#         #     transform=trans,
#         #     label="Noise-optimal" if i == 0 else None,
#         # )
#         ax.scatter(
#             flight.longitude,
#             flight.latitude,
#             color="k",
#             s=2,
#             # linestyle="dashed",
#             transform=trans,
#         )
#         ax.scatter(
#             flight0.longitude,
#             flight0.latitude,
#             color="r",
#             s=2,
#             # linestyle="dashed",
#             transform=trans,
#             label=f"Fuel-optimal" if i == 0 else None,
#         )
#     plt.title(f"{map_type}, max_fuel {max_fuel}")
#     plt.legend()
#     plt.tight_layout()
#     # if max_fuel:
#     #     plt.savefig(f"figs/frn_{map_type}_max_fuel.png", bbox_inches="tight")
#     # else:
#     #     plt.savefig(f"figs/frn_{map_type}.png", bbox_inches="tight")
#     plt.show()
# ## %%
# # %%


# %%
