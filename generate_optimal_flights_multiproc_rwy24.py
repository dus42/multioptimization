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
    rwy = "18L"
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

    c_ends = pd.read_csv(
        f"data_generated/opensky_runway{rwy}_ends_{map_type}.csv"
    ).query("fid==@fid")
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
    optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)
    flight_fuel = optimizer.trajectory(
        objective="fuel",
        h_end=h_end,
        runway_dir=trk_start,
    )
    if flight_fuel is None:
        m0 = 0.999 * m0
        optimizer = top.Climb(actype, start, end, m0=m0)
        optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)
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
    thrust = D + flight_fuel.mass * 9.81 * np.sin(gamma)
    flight_fuel = flight_fuel.assign(thrust=thrust)

    # generate optimalpopulation exposure trajectory
    if map_type == "N" and is_max_fuel:
        max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.02
    elif not is_max_fuel:
        max_fuel = None
    else:
        max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.005

    optimizer = top.Climb(actype, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)

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
        m0 = 0.999 * m0
        optimizer = top.Climb(actype, start, end, m0=m0)
        optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)
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
    flight_pop = flight_pop.assign(cost_grid=cost.full()[0], max_fuel=is_max_fuel)
    flight_fuel = flight_fuel.assign(cost_grid=cost0.full()[0], max_fuel=is_max_fuel)

    return pd.concat([flight_fuel, flight_pop])


# %%
import click


@click.command()
@click.option("--workers", default=6)
def main(workers):
    rwy = "18L"

    for map_type in ["D", "N", "DN"]:
        # for map_type in ["DN"]:
        c_ends = pd.read_csv(f"data_generated/opensky_runway{rwy}_ends_{map_type}.csv")
        options = itertools.product(c_ends.fid.values, [map_type], [True, False])
        options = (
            pd.DataFrame(options, columns=["fid", "map_type", "is_max_fuel"])
            .sort_values(by=["is_max_fuel", "map_type"])
            .reset_index(drop=True)
        )

        flights = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
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
            f"data_generated/optimal_flights_runway{rwy}_{map_type}.csv", index=False
        )


# %%

# %%
if __name__ == "__main__":
    main()
