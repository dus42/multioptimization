# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import casadi
from openap.casadi import aero as aero_casadi
from openap import aero, nav, top, prop
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings
import itertools

import openap
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.colors as mcolors
from traffic.data import airports

pd.set_option("display.max_rows", 15)
warnings.filterwarnings("ignore")


# %%
def _pop_exposure_obj(x, u, dt, optimizer, weight, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = casadi.atan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return weight * cost * thrust * dt + fuel


def obj_pop_exposure_day(x, u, dt, optimizer=None, **kwargs):
    return _pop_exposure_obj(x, u, dt, optimizer, 0.0001)


def obj_pop_exposure_night(x, u, dt, optimizer=None, **kwargs):
    return _pop_exposure_obj(x, u, dt, optimizer, 0.0009)


def obj_max_fuel(x, u, dt, optimizer=None, **kwargs):
    return _pop_exposure_obj(x, u, dt, optimizer, 0.001)


# %%
def opt_trajs(fid, map_type, is_max_fuel=False):
    flight_real = pd.read_parquet(
        f"data_generated/opensky2024_centroids_{map_type}.parquet"
    ).query("flight_id==@fid")
    actype = "a320"
    nodes = 199

    # Determine which population objective(s) to run.
    # DN mode runs both day and night, producing 4 trajectories total.
    if is_max_fuel:
        pop_objectives = [("pop", obj_max_fuel)]
    elif map_type == "DN":
        pop_objectives = [
            ("pop_day", obj_pop_exposure_day),
            ("pop_night", obj_pop_exposure_night),
        ]
    elif map_type == "D":
        pop_objectives = [("pop", obj_pop_exposure_day)]
    else:  # N
        pop_objectives = [("pop", obj_pop_exposure_night)]

    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv").query(
        "fid==@fid"
    )

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

    # ── Fuel-optimal trajectory ───────────────────────────────
    optimizer = top.Climb(actype, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=5000, debug=False)
    flight_fuel = optimizer.trajectory(
        objective="fuel", h_end=h_end, runway_dir=trk_start
    )
    if flight_fuel is None:
        m0 = 0.999 * m0
        optimizer = top.Climb(actype, start, end, m0=m0)
        optimizer.setup(nodes=nodes, max_iteration=10000, debug=False)
        flight_fuel = optimizer.trajectory(
            objective="fuel", h_end=h_end, runway_dir=trk_start
        )
    if flight_fuel is None:
        print(fid, "fuel optimization failed")
        return
    flight_fuel = flight_fuel.assign(fid=fid, obj="fuel")

    drag = openap.Drag(actype, wave_drag=True)
    D = drag.clean(
        mass=flight_fuel.mass,
        tas=flight_fuel.tas,
        alt=flight_fuel.altitude,
        vs=flight_fuel.vertical_rate,
    )
    gamma = np.arctan2(flight_fuel.vertical_rate * aero.fpm, flight_fuel.tas * aero.kts)
    flight_fuel = flight_fuel.assign(thrust=D + flight_fuel.mass * 9.81 * np.sin(gamma))

    # ── Real-flight centroid ──────────────────────────────────
    flight_real = flight_real.assign(fid=fid, obj="real")
    drag = openap.Drag(actype, wave_drag=True)
    D = drag.clean(
        mass=flight_real.mass,
        tas=flight_real.groundspeed,
        alt=flight_real.altitude,
        vs=flight_real.vertical_rate,
    )
    gamma = np.arctan2(
        flight_real.vertical_rate * aero.fpm, flight_real.groundspeed * aero.kts
    )
    flight_real = flight_real.assign(thrust=D + flight_real.mass * 9.81 * np.sin(gamma))
    flight_real = flight_real.assign(
        mach=lambda x: aero.tas2mach(x.groundspeed * aero.kts, x.h)
    ).rename(columns={"groundspeed": "tas"})

    # ── Fuel budget for constrained pop-optimal runs ──────────
    if map_type == "N" and is_max_fuel:
        max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.02
    elif is_max_fuel:
        max_fuel = (flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]) * 1.005
    else:
        max_fuel = None

    # ── Load interpolant for this map type ────────────────────
    interpolant = top.tools.load_interpolant(
        path=f"data_generated/cashed_interp_{map_type}.casadi"
    )

    # ── Population-optimal trajectory/trajectories ────────────
    pop_flights = []
    for obj_label, obj_fn in pop_objectives:
        optimizer = top.Climb(actype, start, end, m0=m0)
        optimizer.setup(nodes=nodes, max_iteration=35000, debug=False)
        objective_with_opt = partial(obj_fn, optimizer=optimizer)

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
            objective_with_opt = partial(obj_fn, optimizer=optimizer)
            flight_pop = optimizer.trajectory(
                objective=objective_with_opt,
                interpolant=interpolant,
                h_end=h_end,
                runway_dir=trk_start,
                max_fuel=max_fuel,
                initial_guess=flight_fuel,
            )
        if flight_pop is None:
            print(fid, f"noise optimization failed (obj={obj_label})")
            return

        flight_pop = flight_pop.assign(fid=fid, obj=obj_label)
        drag = openap.Drag(actype, wave_drag=True)
        D = drag.clean(
            mass=flight_pop.mass,
            tas=flight_pop.tas,
            alt=flight_pop.altitude,
            vs=flight_pop.vertical_rate,
        )
        gamma = np.arctan2(
            flight_pop.vertical_rate * aero.fpm, flight_pop.tas * aero.kts
        )
        flight_pop = flight_pop.assign(
            thrust=D + flight_pop.mass * 9.81 * np.sin(gamma)
        )
        pop_flights.append(flight_pop)

    # ── Attach cost-grid values ───────────────────────────────
    def add_cost(flight):
        cost = interpolant(
            np.array([flight.longitude.values, flight.latitude.values, flight.h.values])
        )
        return flight.assign(cost_grid=cost.full()[0], max_fuel=is_max_fuel)

    flight_fuel = add_cost(flight_fuel)
    flight_real = add_cost(flight_real)[
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
    pop_flights = [add_cost(f) for f in pop_flights]

    return pd.concat([flight_fuel, *pop_flights, flight_real])


# %%
eham = nav.airport("EHAM")
actype = "a320"

for map_type in ["DN"]:
    # for map_type in ["D", "N", "DN"]:

    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

    options = itertools.product(c_ends.fid.values, [map_type], [False])
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
        f"data_generated/optimal_flights_{map_type}_199.csv", index=False
    )


# %%
for max_fuel in [False]:
    map_type = "DN"
    flights = pd.read_csv(f"data_generated/optimal_flights_{map_type}_199.csv")
    df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
    df_real = pd.read_parquet(
        f"data_generated/opensky2024_centroids_{map_type}.parquet"
    )
    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

    flights = flights.query("max_fuel==@max_fuel")
    nx, ny, nz = 50, 45, 23
    cost_grid = df_cost.cost.values.reshape(nx, ny, nz)

    proj = ccrs.TransverseMercator(
        central_longitude=eham["lon"], central_latitude=eham["lat"]
    )
    trans = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection=proj))
    ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
    ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)
    ax.set_extent(
        [
            flights.longitude.min() - 0.2,
            flights.longitude.max() + 0.2,
            flights.latitude.min() - 0.2,
            flights.latitude.max() + 0.2,
        ]
    )

    vmax = 0.05 if map_type == "DN" else 0.03
    norm = plt.Normalize(vmin=0.00001, vmax=vmax, clip=True)
    ax.contourf(
        df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 3],
        df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 3],
        cost_grid[:, :, 3],
        cmap="binary",
        transform=trans,
        levels=20,
        norm=norm,
        alpha=0.5,
    )

    # Style config per obj tag — extend here for new trajectory types
    obj_styles = {
        "fuel": dict(color="r", lw=1, linestyle="dashed", label="Fuel-optimal"),
        "pop": dict(color="k", lw=2, linestyle="solid", label="Population-optimal"),
        "pop_day": dict(color="k", lw=2, linestyle="solid", label="Pop-optimal (day)"),
        "pop_night": dict(
            color="purple", lw=2, linestyle="solid", label="Pop-optimal (night)"
        ),
    }

    for i, fid in enumerate(flights.fid.unique()):
        flightr = df_real.query(f"flight_id=='{fid}'")
        ax.plot(
            flightr.longitude,
            flightr.latitude,
            color="tab:blue",
            lw=2,
            transform=trans,
            label="Real flights centroids" if i == 0 else None,
        )
        for obj_tag, style in obj_styles.items():
            seg = flights.query(f"fid==@fid and obj=='{obj_tag}'")
            if seg.empty:
                continue
            ax.plot(
                seg.longitude,
                seg.latitude,
                transform=trans,
                label=style["label"] if i == 0 else None,
                **{k: v for k, v in style.items() if k != "label"},
            )

    plt.legend()
    plt.tight_layout()
    out = f"figs/frn_{map_type}{'_max_fuel' if max_fuel else ''}.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.show()
