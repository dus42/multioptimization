# %%
import warnings
import itertools
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import casadi
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import openap
from openap.casadi import aero as aero_casadi
from openap import aero, nav, top, prop
from traffic.data import airports
from traffic.core import Traffic, Flight

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
ACTYPE = "a320"
NODES = 39
MAX_ITER_FUEL = 5000
MAX_ITER_POP = 35_000

# Weighting coefficients per objective
OBJ_WEIGHTS = {
    "day": 0.0001,
    "night": 0.0009,
    "max_fuel": 0.001,
}


# ──────────────────────────────────────────────────────────────
# Objective functions
# ──────────────────────────────────────────────────────────────
def _pop_exposure_obj(x, u, dt, optimizer, weight: float, **kwargs):
    """Generic population-exposure objective, parameterised by weight."""
    _, _, h, m, _ = x[0], x[1], x[2], x[3], x[4]
    mach, vs, _ = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = casadi.atan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return weight * cost * thrust * dt + fuel


def obj_pop_exposure_day(x, u, dt, optimizer=None, **kwargs):
    return _pop_exposure_obj(x, u, dt, optimizer, OBJ_WEIGHTS["day"], **kwargs)


def obj_pop_exposure_night(x, u, dt, optimizer=None, **kwargs):
    return _pop_exposure_obj(x, u, dt, optimizer, OBJ_WEIGHTS["night"], **kwargs)


def obj_max_fuel(x, u, dt, optimizer=None, **kwargs):
    return _pop_exposure_obj(x, u, dt, optimizer, OBJ_WEIGHTS["max_fuel"], **kwargs)


# Map-type → objective function (for non-DN or non-max-fuel cases)
MAP_OBJ = {
    "D": obj_pop_exposure_day,
    "N": obj_pop_exposure_night,
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _load_interpolant(map_type: str):
    """Load a cached CasADi interpolant for the given map type."""
    return top.tools.load_interpolant(
        path=f"data_generated/cashed_interp_{map_type}.casadi"
    )
    # return top.tools.load_interpolant(
    # path=f"data_generated/cashed_interp_DN_15k.casadi"
    # )


def _add_thrust(flight: pd.DataFrame, tas_col: str = "tas") -> pd.DataFrame:
    """Append a thrust column computed from aerodynamic drag + climb component."""
    drag = openap.Drag(ACTYPE, wave_drag=True)
    D = drag.clean(
        mass=flight.mass,
        tas=flight[tas_col],
        alt=flight.altitude,
        vs=flight.vertical_rate,
    )
    gamma = np.arctan2(
        flight.vertical_rate * aero.fpm,
        flight[tas_col] * aero.kts,
    )
    return flight.assign(thrust=D + flight.mass * 9.81 * np.sin(gamma))


def _make_optimizer(start, end, m0: float, max_iter: int):
    optimizer = top.Climb(ACTYPE, start, end, m0=m0)
    optimizer.setup(nodes=NODES, max_iteration=max_iter, debug=False)
    return optimizer


def _optimize_fuel(start, end, m0: float, h_end: float, trk_start, alt_start):
    """Try fuel-optimal trajectory; retry once with slightly reduced m0."""
    for m0_try, max_iter in [(m0, MAX_ITER_FUEL), (0.999 * m0, 10_000)]:
        opt = _make_optimizer(start, end, m0_try, max_iter)
        result = opt.trajectory(
            objective="fuel", h_end=h_end, runway_dir=trk_start, alt_start=alt_start
        )
        if result is not None:
            return result, m0_try
    return None, m0


def _optimize_pop(
    start,
    end,
    m0: float,
    h_end: float,
    trk_start,
    obj_fn,
    interpolant,
    max_fuel,
    initial_guess,
    alt_start,
):
    """Try population-optimal trajectory; retry once with slightly reduced m0."""
    for m0_try, max_iter in [(m0, MAX_ITER_POP), (0.99 * m0, MAX_ITER_POP)]:
        opt = _make_optimizer(start, end, m0_try, max_iter)
        objective_with_opt = partial(obj_fn, optimizer=opt)
        result = opt.trajectory(
            objective=objective_with_opt,
            interpolant=interpolant,
            h_end=h_end,
            runway_dir=trk_start,
            max_fuel=max_fuel,
            initial_guess=initial_guess,
            alt_start=alt_start,
        )
        if result is not None:
            return result
    return None


def _assign_cost_grid(flight: pd.DataFrame, interpolant) -> pd.DataFrame:
    cost = interpolant(
        np.array([flight.longitude.values, flight.latitude.values, flight.h.values])
    )
    return flight.assign(cost_grid=cost.full()[0])


REAL_FLIGHT_COLS = [
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


# ──────────────────────────────────────────────────────────────
# Core per-flight optimisation
# ──────────────────────────────────────────────────────────────
def opt_trajs(fid: str, map_type: str, is_max_fuel: bool = False):
    """
    Optimise departure trajectories for a single flight cluster centroid.

    For map_type "DN", returns 4 trajectories:
        fuel-optimal, real, pop-optimal (day), pop-optimal (night).
    For "D" / "N", returns 3 trajectories:
        fuel-optimal, real, pop-optimal (day or night respectively).

    Returns a concatenated DataFrame, or None on failure.
    """
    # ── Load real-flight centroid ──────────────────────────────
    flight_real = (
        Flight(
            pd.read_parquet(
                f"data_generated/opensky2024_centroids_{map_type}.parquet"
            ).query("flight_id==@fid")
        )
        .resample(NODES + 1)
        .data
    )
    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv").query(
        "fid==@fid"
    )

    # ── Runway / start geometry ───────────────────────────────
    rwy = airports["EHAM"].runways.data.query(f"name=='{c_ends.runway.values[0]}'")
    if rwy is None or len(rwy) == 0:
        trk_start = None
        start = "EHAM"
    else:
        trk_start = rwy.bearing.values[0]
        start = (rwy.latitude.values[0], rwy.longitude.values[0])

    end = (c_ends.latitude.values[0], c_ends.longitude.values[0])
    h_end = c_ends.altitude.values[0] * aero.ft
    alt_start = c_ends.alt_start.values[0]
    m0 = c_ends.tow.values[0] / prop.aircraft(ACTYPE)["mtow"]
    if m0 > 0.95:
        m0 *= 0.99

    # ── Fuel-optimal trajectory ───────────────────────────────
    flight_fuel_raw, m0 = _optimize_fuel(
        start, end, m0, h_end, trk_start, alt_start=alt_start
    )
    if flight_fuel_raw is None:
        print(f"{fid}: fuel optimisation failed")
        return None

    flight_fuel = _add_thrust(flight_fuel_raw).assign(fid=fid, obj="fuel")

    # ── Real-flight centroid ──────────────────────────────────
    flight_real = (
        _add_thrust(flight_real, tas_col="groundspeed")
        .assign(
            fid=fid,
            obj="real",
            mach=lambda df: aero.tas2mach(df.groundspeed * aero.kts, df.h),
        )
        .rename(columns={"groundspeed": "tas"})
    )

    # ── Fuel budget for constrained pop-optimal runs ──────────
    fuel_used = flight_fuel.mass.values[0] - flight_fuel.mass.values[-1]
    if map_type == "N" and is_max_fuel:
        max_fuel = fuel_used * 1.02
    elif is_max_fuel:
        max_fuel = fuel_used * 1.005
    else:
        max_fuel = None

    # ── Determine which objective(s) to run ───────────────────
    if is_max_fuel:
        obj_fns = [("pop", obj_max_fuel)]
    elif map_type == "DN":
        # Four-trajectory mode: one pop run per time-of-day weighting
        obj_fns = [
            ("pop_day", obj_pop_exposure_day),
            ("pop_night", obj_pop_exposure_night),
        ]
    else:
        obj_fns = [("pop", MAP_OBJ[map_type])]

    # ── Load interpolant(s) ────────────────────────────────────
    # For DN we have a single combined interpolant; D/N have their own.
    interpolant = _load_interpolant(map_type)

    # ── Run population-optimal optimisations ──────────────────
    pop_flights = []
    for obj_label, obj_fn in obj_fns:
        flight_pop_raw = _optimize_pop(
            start,
            end,
            m0,
            h_end,
            trk_start,
            obj_fn,
            interpolant,
            max_fuel,
            initial_guess=flight_fuel,
            alt_start=alt_start,
        )
        if flight_pop_raw is None:
            print(f"{fid}: noise/pop optimisation failed (obj={obj_label})")
            return None
        flight_pop = _add_thrust(flight_pop_raw).assign(fid=fid, obj=obj_label)
        pop_flights.append(flight_pop)

    # ── Attach cost-grid values ───────────────────────────────
    flight_fuel = _assign_cost_grid(flight_fuel, interpolant).assign(
        max_fuel=is_max_fuel
    )
    flight_real = _assign_cost_grid(flight_real, interpolant).assign(
        max_fuel=is_max_fuel
    )[REAL_FLIGHT_COLS]
    pop_flights = [
        _assign_cost_grid(f, interpolant).assign(max_fuel=is_max_fuel)
        for f in pop_flights
    ]

    return pd.concat([flight_fuel, *pop_flights, flight_real], ignore_index=True)


# ──────────────────────────────────────────────────────────────
# Batch execution
# ──────────────────────────────────────────────────────────────
def run_batch(map_type: str, is_max_fuel: bool = False, max_workers: int = 5):
    c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
    fids = c_ends.fid.values

    flights = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = {
            executor.submit(opt_trajs, fid, map_type, is_max_fuel): fid for fid in fids
        }
        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            ncols=0,
            desc=f"Optimising [{map_type}, max_fuel={is_max_fuel}]",
        ):
            result = future.result()
            if result is not None:
                flights.append(result)

    if not flights:
        print(f"No results for map_type={map_type}")
        return

    suffix = "_max_fuel" if is_max_fuel else ""
    out_path = f"data_generated/optimal_flights_{map_type}{suffix}_{NODES}.csv"
    pd.concat(flights, ignore_index=True).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────
# Label / style config per objective tag
OBJ_STYLE = {
    "fuel": dict(color="tab:green", lw=2, linestyle="solid", label="Fuel-optimal"),
    "pop": dict(color="k", lw=2, linestyle="solid", label="Pop-optimal"),
    "pop_day": dict(
        color="tab:orange",
        lw=2,
        linestyle="solid",
        label=r"Population-optimal ($c_{dn}=0.1 × 10^{-3}$)",
    ),
    "pop_night": dict(
        color="tab:blue",
        lw=2,
        linestyle="solid",
        label=r"Population-optimal ($c_{dn}=0.9 × 10^{-3}$)",
    ),
    "real": dict(
        color="darkred", lw=1, linestyle="dashed", label="Real flight centroids"
    ),
}


def _add_compass(ax, x=0.06, y=0.06, size=0.045):
    """
    Add a minimal N/S/E/W compass rose to a cartopy GeoAxes.
    x, y : axes-fraction position of the compass centre (default: top-right)
    size  : radius of the compass in axes-fraction units
    """
    arrows = {
        "N": (0, 1, "N"),
        "S": (0, -1, "S"),
        "E": (1, 0, "E"),
        "W": (-1, 0, "W"),
    }
    ax_to_disp = ax.transAxes  # axes → display
    disp_to_ax = ax.transAxes.inverted()

    # Convert centre and a unit step to display coords so the compass
    # stays visually square regardless of the axes aspect ratio
    cx_d, cy_d = ax_to_disp.transform((x, y))
    # 1 % of axes width/height in display units
    ux = ax_to_disp.transform((x + size, y))[0] - cx_d
    uy = ax_to_disp.transform((x, y + size))[1] - cy_d

    for dx, dy, label in arrows.values():
        # Tip of the arrow (display coords)
        tx, ty = cx_d - dx * ux, cy_d - dy * uy
        # Convert tip and centre back to axes fraction for annotate
        tax, tay = disp_to_ax.transform((tx, ty))
        annotate_kw = dict(
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            color="k",
            annotation_clip=False,
        )
        # Draw arrow from centre → tip, label just beyond the tip
        label_x = x + dx * size * 1.65
        label_y = y + dy * size * 1.65
        ax.annotate(
            label,
            xy=(tax, tay),  # arrow head
            xytext=(label_x, label_y),  # label / arrow tail
            **annotate_kw,
        )

    # Small filled circle at the compass centre
    ax.plot(
        x,
        y,
        "o",
        color="k",
        markersize=3,
        transform=ax.transAxes,
        zorder=5,
        clip_on=False,
    )


def plot_trajectories(map_type: str, is_max_fuel: bool = False):
    eham = nav.airport("EHAM")
    suffix = "_max_fuel" if is_max_fuel else ""
    flights = pd.read_csv(
        f"data_generated/optimal_flights_{map_type}{suffix}_{NODES}.csv"
    )
    df_cost = pd.read_parquet(f"data_generated/df_cost_{map_type}_ext.parquet")
    df_real = pd.read_parquet(
        f"data_generated/opensky2024_centroids_{map_type}.parquet"
    )
    flights = flights.query("max_fuel==@is_max_fuel")
    nx, ny, nz = 100, 100, 50
    cost_grid = df_cost.cost.values.reshape(nx, ny, nz)
    proj = ccrs.TransverseMercator(
        central_longitude=eham["lon"], central_latitude=eham["lat"]
    )
    trans = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(projection=proj))
    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gl.ylocator = mticker.FixedLocator([51, 53])
    gl.xlocator = mticker.FixedLocator([2, 4, 6])
    gl.bottom_labels = True
    gl.right_labels = True
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
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
    norm = plt.Normalize(vmin=1e-5, vmax=vmax, clip=True)
    ax.contourf(
        df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 3],
        df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 3],
        cost_grid[:, :, 3],
        cmap="binary",
        transform=trans,
        levels=20,
        norm=norm,
        alpha=0.3,
    )
    legend_added = set()
    for fid in flights.fid.unique():
        # Real flight
        flightr = df_real.query(f"flight_id=='{fid}'")
        style = OBJ_STYLE["real"]
        ax.plot(
            flightr.longitude,
            flightr.latitude,
            transform=trans,
            label=style["label"] if "real" not in legend_added else None,
            **{k: v for k, v in style.items() if k != "label"},
        )
        legend_added.add("real")
        # Optimised trajectories
        for obj_tag, style in OBJ_STYLE.items():
            if obj_tag == "real":
                continue
            seg = flights.query(f"fid==@fid and obj=='{obj_tag}'")
            if seg.empty:
                continue
            ax.plot(
                seg.longitude,
                seg.latitude,
                transform=trans,
                label=style["label"] if obj_tag not in legend_added else None,
                **{k: v for k, v in style.items() if k != "label"},
            )
            legend_added.add(obj_tag)

    _add_compass(ax, x=0.1, y=0.1, size=0.025)  # ← compass rose

    plt.legend()
    plt.tight_layout()
    out = f"figs/frn_{map_type}{suffix}_{NODES}.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Figure saved → {out}")


# %%
# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for map_type in ["DN"]:  # change to ["D", "N", "DN"] to run all
        for is_max_fuel in [False]:  # add True to also run fuel-capped variants
            run_batch(map_type, is_max_fuel=is_max_fuel)
            plot_trajectories(map_type, is_max_fuel=is_max_fuel)


# %%
