# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=4, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.001 * cost * thrust * dt + fuel


def obj_pop_exposure_night(x, u, dt, optimizer=None, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = np.arctan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=4, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.0009 * cost * thrust * dt + fuel


def obj_max_fuel(x, u, dt, optimizer=None, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = np.arctan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=4, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.001 * cost * thrust * dt + fuel


# %%


def opt_trajs(fid, map_type, is_max_fuel=False):
    flight_real = pd.read_parquet(
        f"data_generated/opensky2024_centroids_{map_type}.parquet"
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
    # month = c_ends.month.values[0]
    # month = f"{month:02d}"
    # if map_type == "DN":
    #     month = ""
    # df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}{month}.csv")
    # if map_type != "DN":
    #     df_cost["cost"] = df_cost["cost"].apply(lambda x: x * 2)

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
    optimizer.setup(nodes=nodes, max_iteration=5000, debug=False)
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
    # print(fid, "fuel is done")
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

    # interpolant = top.tools.interpolant_from_dataframe(df_cost)
    # interpolant = top.tools.load_interpolant(
    #     path=f"data_generated/cashed_interp_{map_type}.casadi"
    # )
    interpolant = top.tools.load_interpolant(
        path=f"data_generated/cashed_interp_transition_reverse.casadi"
    )
    # df = pd.read_parquet("data_generated/df_cost_D_ext.parquet")
    # interpolant = top.tools.interpolant_from_dataframe(df, shape="linear")
    #
    flight_pop = optimizer.trajectory(
        objective=objective_with_opt,
        interpolant=interpolant,
        h_end=h_end,
        runway_dir=trk_start,
        max_fuel=max_fuel,
        # initial_guess=flight_fuel,
    )
    if flight_pop is None:
        m0 = 0.99 * m0
        optimizer = top.Climb(actype, start, end, m0=m0)
        optimizer.setup(nodes=nodes, max_iteration=35000, debug=False)

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
    interpolant = top.tools.load_interpolant(
        path=f"data_generated/cashed_interp_{map_type}.casadi"
    )
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
    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv").query(
        "fid=='VLG53KX_284'"
    )

    # options = itertools.product(c_ends.fid.values, [map_type], [True, False])
    options = itertools.product(c_ends.fid.values[:], [map_type], [False])
    options = (
        pd.DataFrame(options, columns=["fid", "map_type", "is_max_fuel"])
        .sort_values(by=["is_max_fuel", "map_type"])
        .reset_index(drop=True)
    )

    flights = []

    with ProcessPoolExecutor(max_workers=5) as executor:
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
        f"data_generated/optimal_flights_DN_transition_reverse.csv", index=False
    )


# %%
eham = nav.airport("EHAM")
actype = "a320"
start = (eham["lat"], eham["lon"])
nodes = 99


import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.colors as mcolors


for max_fuel in [False, True][:1]:
    map_type = "DN"
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors.extend(["b", "g", "y", "m", "c"])
    flights = pd.read_csv(f"data_generated/optimal_flights_{map_type}_linear.csv")
    df_cost = pd.read_parquet(f"data_generated/df_cost_{map_type}_ext.parquet")
    df_real = pd.read_parquet(
        f"data_generated/opensky2024_centroids_{map_type}.parquet"
    )

    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

    flights = flights.query("max_fuel==@max_fuel")
    nx, ny, nz = 100, 100, 50
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
    for i, fid in enumerate(flights.query("fid=='VLG53KX_284'").fid.unique()):
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
        plt.savefig(f"figs/frn_{map_type}_{nodes}.png", bbox_inches="tight", dpi=300)
    plt.show()
## %%
# %%
eham = nav.airport("EHAM")
actype = "a320"
start = (eham["lat"], eham["lon"])
nodes = 39


import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.colors as mcolors


for max_fuel in [False, True][:1]:
    map_type = "DN"
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors.extend(["b", "g", "y", "m", "c"])
    f_d = (
        pd.read_csv(f"data_generated/optimal_flights_D_transition.csv")
        .query("obj=='pop'")
        .assign(mapp="day")
    )
    f_n = (
        pd.read_csv(f"data_generated/optimal_flights_N_transition.csv")
        .query("obj=='pop'")
        .assign(mapp="night")
    )
    f_t = (
        pd.read_csv(f"data_generated/optimal_flights_DN_transition.csv")
        .query("obj=='pop'")
        .assign(mapp="trans")
    )
    f_tr = (
        pd.read_csv(f"data_generated/optimal_flights_DN_transition_reverse.csv")
        .query("obj=='pop'")
        .assign(mapp="trans")
    )
    df_cost_d = pd.read_parquet(f"data_generated/df_cost_D_ext.parquet")
    df_cost_n = pd.read_parquet(f"data_generated/df_cost_N_ext.parquet")
    # df_real = pd.read_parquet(
    #     f"data_generated/opensky2024_centroids_{map_type}.parquet"
    # )

    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

    nx, ny, nz = 100, 100, 50

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
            min(f_t.longitude.values) - 0.2,
            max(f_t.longitude.values) + 0.2,
            min(f_t.latitude.values) - 0.2,
            max(f_t.latitude.values) + 0.2,
        ]
    )
    ax.set_extent(
        [
            min(f_t.query("fid=='VLG53KX_284'").longitude.values) - 0.4,
            max(f_t.query("fid=='VLG53KX_284'").longitude.values) + 0.4,
            min(f_t.query("fid=='VLG53KX_284'").latitude.values) + 0.6,
            max(f_t.query("fid=='VLG53KX_284'").latitude.values) + 0.4,
        ]
    )

    norm = plt.Normalize(vmin=0.00001, vmax=1, clip=True)
    if map_type == "DN":
        norm = plt.Normalize(vmin=0.00001, vmax=0.05, clip=True)
    cntr = ax.contourf(
        df_cost_d.longitude.values.reshape(nx, ny, nz)[:, :, 3],
        df_cost_d.latitude.values.reshape(nx, ny, nz)[:, :, 3],
        df_cost_d.cost.values.reshape(nx, ny, nz)[:, :, 3],
        # cmap="PuRd",
        cmap="Blues",
        transform=trans,
        levels=50,
        norm=norm,
        alpha=0.5,
    )
    cntr = ax.contourf(
        df_cost_n.longitude.values.reshape(nx, ny, nz)[:, :, 3],
        df_cost_n.latitude.values.reshape(nx, ny, nz)[:, :, 3],
        df_cost_n.cost.values.reshape(nx, ny, nz)[:, :, 3],
        cmap="binary",
        transform=trans,
        levels=20,
        norm=norm,
        alpha=0.5,
    )
    # xx=0
    # for i, fid in enumerate(flights.fid.unique()[xx:xx+1]):
    for i, fid in enumerate(f_t.query("fid=='VLG53KX_284'").fid.unique()[:]):
        # flightr = df_real.query(f"flight_id=='{fid}'")
        # ax.plot(
        #     flightr.longitude,
        #     flightr.latitude,
        #     color="tab:blue",
        #     lw=2,
        #     transform=trans,
        #     label="Real flights centroids" if i == 0 else None,
        # )
        t = f_t.query("fid==@fid")
        tr = f_tr.query("fid==@fid")
        d = f_d.query("fid==@fid")
        n = f_n.query("fid==@fid")
        print("t", t.ts.max())
        print("tr", tr.ts.max())
        print("d", d.ts.max())
        print("n", n.ts.max())
        # ax.plot(
        #     t.query("cost_grid>0").longitude,
        #     t.query("cost_grid>0").latitude,
        #     color="k",
        #     lw=2,
        #     transform=trans,
        #     label="Population-optimal" if i == 0 else None,
        # )

        ax.plot(
            d.longitude,
            d.latitude,
            color="darkorchid",
            lw=3,
            # linestyle="dashed",
            transform=trans,
            label=f"Day map" if i == 0 else None,
            # alpha=0.7,
        )
        ax.plot(
            n.longitude,
            n.latitude,
            color="k",
            lw=3,
            # linestyle="dashed",
            transform=trans,
            label=f"Night map" if i == 0 else None,
            # alpha=0.7,
        )
        ax.plot(
            t.longitude,
            t.latitude,
            color="darkorange",
            lw=3,
            # linestyle="dashed",
            transform=trans,
            label=f"Transition" if i == 0 else None,
            # alpha=0.7,
        )
        ax.plot(
            tr.longitude,
            tr.latitude,
            color="green",
            lw=3,
            # linestyle="dashed",
            transform=trans,
            label=f"Transition" if i == 0 else None,
            # alpha=0.7,
        )
    # plt.title(f"{map_type}, max_fuel {max_fuel}")
    plt.legend()
    plt.tight_layout()
    if max_fuel:
        plt.savefig(f"figs/frn_{map_type}_max_fuel.png", bbox_inches="tight")
    else:
        plt.savefig(f"figs/transition_All.png", bbox_inches="tight", dpi=300)
    plt.show()
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
import pandas as pd
import numpy as np

df_cost = pd.read_csv(f"../data_generated/df_cost_D_new.csv")
# Current altitude spacing
alt_spacing = np.sort(df_cost["altitude"].unique())
dz = alt_spacing[1] - alt_spacing[0]

# Existing max altitude
alt_max = df_cost["altitude"].max()

# Target altitude (45 kft)
target_alt = 50000  # assuming altitude is in ft

# New altitude levels
new_alts = np.arange(alt_max + dz, target_alt + dz, dz)

# Unique lon/lat mesh
lonlat = df_cost[["longitude", "latitude"]].drop_duplicates()

# Create new rows
new_rows = []

for alt in new_alts:
    temp = lonlat.copy()
    temp["altitude"] = alt
    temp["cost"] = 0
    new_rows.append(temp)

# Combine all new altitude layers
df_new = pd.concat(new_rows, ignore_index=True)

# Append to original dataframe
df_cost_extended = pd.concat([df_cost, df_new], ignore_index=True)

# Optional: sort
df_cost_extended = df_cost_extended.sort_values(
    ["longitude", "latitude", "altitude"]
).reset_index(drop=True)
df_cost_extended = df_cost_extended.assign(height=lambda x: x.altitude * aero.ft)
df_cost_extended.to_parquet(f"../data_generated/df_cost_D_ext.parquet")
interpolant = top.tools.interpolant_from_dataframe(df_cost_extended, shape="bspline")
top.tools.save_interpolant(
    interpolant=interpolant, path=f"../data_generated/cashed_interp_D.casadi"
)
# %%
df = pd.read_parquet("data_generated/df_cost_DN_ext.parquet")
# %%
df.loc[df.query("altitude>15000").index, "cost"] = 0
df.to_parquet("data_generated/df_cost_DN_ext_15k.parquet")
interpolant = top.tools.interpolant_from_dataframe(df, shape="bspline")
top.tools.save_interpolant(
    interpolant=interpolant, path=f"data_generated/cashed_interp_DN_15k.casadi"
)
# %% Transition map
d = pd.read_parquet("data_generated/df_cost_D_ext.parquet").assign(ts=900)
d["longitude"] = d["longitude"].round(6)
d["latitude"] = d["latitude"].round(6)
d["altitude"] = d["altitude"].round(6)


n = pd.read_parquet("data_generated/df_cost_N_ext.parquet").assign(ts=0)
n["latitude"] = n["latitude"].round(6)
n["altitude"] = n["altitude"].round(6)
n["longitude"] = n["longitude"].round(6)
df = (
    pd.concat([d, n])
    .assign(height=lambda x: x.altitude * aero.ft)
    .sort_values(by=["ts", "height", "latitude", "longitude"], ascending=True)
)

interpolant = top.tools.interpolant_from_dataframe(df, shape="linear")
top.tools.save_interpolant(
    interpolant=interpolant,
    path=f"data_generated/cashed_interp_transition_reverse.casadi",
)

# %%
# %%
for max_fuel in [False, True][:1]:
    map_type = "DN"
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors.extend(["b", "g", "y", "m", "c"])
    f_d = (
        pd.read_csv(f"data_generated/optimal_flights_D_transition.csv")
        .query("obj=='pop'")
        .assign(mapp="day")
    )
    f_n = (
        pd.read_csv(f"data_generated/optimal_flights_N_transition.csv")
        .query("obj=='pop'")
        .assign(mapp="night")
    )
    f_t = (
        pd.read_csv(f"data_generated/optimal_flights_DN_transition.csv")
        .query("obj=='pop'")
        .assign(mapp="trans")
    )
    f_tr = (
        pd.read_csv(f"data_generated/optimal_flights_DN_transition_reverse.csv")
        .query("obj=='pop'")
        .assign(mapp="trans")
    )
    df_cost_d = pd.read_parquet(f"data_generated/df_cost_D_ext.parquet")
    df_cost_n = pd.read_parquet(f"data_generated/df_cost_N_ext.parquet")

    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

    nx, ny, nz = 100, 100, 50

    proj = ccrs.TransverseMercator(
        central_longitude=eham["lon"], central_latitude=eham["lat"]
    )
    trans = ccrs.PlateCarree()

    # --- Compute shared extent ---
    fid_query = "VLG53KX_284"
    extent = [
        min(f_t.query(f"fid=='{fid_query}'").longitude.values) - 0.4,
        max(f_t.query(f"fid=='{fid_query}'").longitude.values) + 0.4,
        min(f_t.query(f"fid=='{fid_query}'").latitude.values) + 0.6,
        max(f_t.query(f"fid=='{fid_query}'").latitude.values) + 0.4,
    ]

    norm = plt.Normalize(vmin=0.00001, vmax=0.05, clip=True)

    # --- Layout: 4 small subplots on left, 1 large on right ---
    fig = plt.figure(figsize=(6, 7))

    # GridSpec: 4 rows, 2 cols — left col narrow (subplots), right col wide (main)
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(4, 2, figure=fig, width_ratios=[1, 4], hspace=0.3, wspace=0.05)

    ax_main = fig.add_subplot(gs[:, 1], projection=proj)

    sub_axes = [fig.add_subplot(gs[i, 0], projection=proj) for i in range(4)]

    # --- Helper to add background contours to any axis ---
    def add_background_d(ax):
        ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
        ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)
        ax.set_extent(extent)
        ax.contourf(
            df_cost_d.longitude.values.reshape(nx, ny, nz)[:, :, 3],
            df_cost_d.latitude.values.reshape(nx, ny, nz)[:, :, 3],
            df_cost_d.cost.values.reshape(nx, ny, nz)[:, :, 3],
            cmap="Blues",
            transform=trans,
            levels=20,
            norm=norm,
            alpha=0.3,
        )

    def add_background_n(ax):
        ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
        ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)
        ax.set_extent(extent)

        ax.contourf(
            df_cost_n.longitude.values.reshape(nx, ny, nz)[:, :, 3],
            df_cost_n.latitude.values.reshape(nx, ny, nz)[:, :, 3],
            df_cost_n.cost.values.reshape(nx, ny, nz)[:, :, 3],
            cmap="binary",
            transform=trans,
            levels=20,
            norm=norm,
            alpha=0.3,
        )

    # --- Trajectory definitions ---
    trajectories = [
        (f_d, "darkorchid", "Day map"),
        (f_n, "k", "Night map"),
        (f_t, "darkorange", "Day --> Night"),
        (f_tr, "green", "Night --> Day"),
    ]

    # --- Draw main plot (all 4 overlaid) ---
    add_background_d(ax_main)
    add_background_n(ax_main)
    for i, fid in enumerate(f_t.query(f"fid=='{fid_query}'").fid.unique()):
        t = f_t.query("fid==@fid")
        tr = f_tr.query("fid==@fid")
        d = f_d.query("fid==@fid")
        n = f_n.query("fid==@fid")

        for df, color, label in trajectories:
            subset = df.query("fid==@fid")
            ax_main.plot(
                subset.longitude,
                subset.latitude,
                color=color,
                lw=3,
                transform=trans,
                label=label if i == 0 else None,
            )

    ax_main.legend(fontsize=10, loc="lower right")
    gl = ax_main.gridlines(
        draw_labels=False,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.bottom_labels = True
    gl.right_labels = True
    gl.ylocator = mticker.FixedLocator([51, 51.5, 52, 52.5])
    gl.xlocator = mticker.FixedLocator([3.5, 4, 4.5, 5, 5.5])

    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    # ax_main.set_title("All trajectories", fontsize=12)

    # --- Draw individual subplots ---
    sub_titles = ["Static Day", "Static Night", "Day --> Night", "Night --> Day"]
    sub_colors = ["darkorchid", "k", "darkorange", "green"]
    sub_dfs = [f_d, f_n, f_t, f_tr]

    for ax_sub, df_sub, color, title in zip(sub_axes, sub_dfs, sub_colors, sub_titles):

        if title in ["Static Day", "Day --> Night"]:
            add_background_d(ax_sub)
        else:
            add_background_n(ax_sub)
        for fid in f_t.query(f"fid=='{fid_query}'").fid.unique():
            subset = df_sub.query("fid==@fid")
            ax_sub.plot(
                subset.longitude,
                subset.latitude,
                color=color,
                lw=2,
                transform=trans,
            )
        ax_sub.set_extent([4, 5.2, 51.8, 52.5])
        ax_sub.set_title(title, fontsize=9)

    # plt.suptitle(f"{map_type} – Population-optimal trajectories", fontsize=11, y=1.01)
    plt.tight_layout()

    if max_fuel:
        plt.savefig(f"figs/frn_{map_type}_max_fuel.png", bbox_inches="tight")
    else:
        plt.savefig(f"figs/transition_All.png", bbox_inches="tight", dpi=300)
    plt.show()

# %%
