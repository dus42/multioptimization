# %%
from traffic.core import Traffic, Flight
import pandas as pd
import numpy as np
from openap import FuelFlow, Thrust, aero, top
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer, CRS
import itertools
from tqdm import tqdm
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
# %%


# Load once at module level
_npdc = None


def _get_npdc():
    global _npdc
    if _npdc is None:
        npdc = pd.read_csv(curr_path + "/../data_raw/npds_check.csv").query(
            "noise_metric=='LAmax'"
        )
        thrusts = npdc.thrust_n.to_numpy()
        dists = np.array([int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]])
        noise_levels = npdc.iloc[:, 5:15].values
        _npdc = (thrusts, dists, noise_levels)
    return _npdc


def interp_npd1(thrs, dis):
    thrusts, dists, noise_levels = _get_npdc()

    # Clamp distances and thrusts to valid range
    d_clamped = np.clip(dis, dists[0], dists[-2])  # keep below max so d2 is valid
    t_clamped = np.clip(thrs, thrusts[0], thrusts[-2])

    # Find lower bracket indices via searchsorted
    d_idx = np.searchsorted(dists, d_clamped, side="right") - 1
    d_idx = np.clip(d_idx, 0, len(dists) - 2)
    t_idx = np.searchsorted(thrusts, t_clamped, side="right") - 1
    t_idx = np.clip(t_idx, 0, len(thrusts) - 2)

    d1 = dists[d_idx]
    d2 = dists[d_idx + 1]
    p1_idx = t_idx
    p2_idx = t_idx + 1

    L_p1_d1 = noise_levels[p1_idx, d_idx]
    L_p1_d2 = noise_levels[p1_idx, d_idx + 1]
    L_p2_d1 = noise_levels[p2_idx, d_idx]
    L_p2_d2 = noise_levels[p2_idx, d_idx + 1]

    # Log-linear interpolation in distance
    log_d = np.log10(np.clip(dis, dists[0], dists[-1]))
    log_d1 = np.log10(d1)
    log_d2 = np.log10(d2)
    w_d = (log_d - log_d1) / (log_d2 - log_d1)

    L_p1_d = L_p1_d1 + (L_p1_d2 - L_p1_d1) * w_d
    L_p2_d = L_p2_d1 + (L_p2_d2 - L_p2_d1) * w_d

    # Linear interpolation in thrust
    p1 = thrusts[p1_idx]
    p2 = thrusts[p2_idx]
    w_t = (thrs - p1) / (p2 - p1)
    noise = L_p1_d + (L_p2_d - L_p1_d) * w_t

    noise[dis >= dists.max()] = 0

    return noise


def pp_affected(ac, pop_map, grid_type, flight, treshold):
    """Calculate people affected by flight noise exceeding the indicated threshold

    Args:
        ac (string): ICAO aircraft type (for example: A320).
        pop_map: pandas dsataframe of population density map
        grid_type (string): "xy" or "ll" type of grid of the population map
        flight: pandas dataframe of a flight trajectory required columns named as follows: ts, tas, mass, altitude, vertical_rate, longitude, latitude
                where   ts is timestamps in seconds
                        tas is TAS in knots
                        mass in kg
                        altitude in feet
                        vertical_rate in feet per minute
        treshold (int): treshold of LAmax db(A)
        returns:
        pp_affected_all (int): sum of people affected by noise exceeding the indicated treshold at all timestamps
        pp_affected_unique (int): sum of unique people affected by noise exceeding the indicated treshold
        flight: a pandas dataframe of the input fligth with two additional columns : thrust, fuel
                where   thrust in N is estimated thrust at the given instance of timestamp
                        fuel in kg is estimated fuel spent along the route
    """
    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
    transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)
    # interp_npd = get_npd_interpolator(ac)
    xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
    point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
    min_x, min_y = (xf.min() - 8000, yf.min() - 8000)
    max_x, max_y = (xf.max() + 8000, yf.max() + 8000)
    if grid_type == "xy":
        pm = pop_map.query("@min_x<x<@max_x and @min_y<y<@max_y").copy()
        pop_x = np.sort(pm.x.unique())
        pop_y = np.sort(pm.y.unique())
        grid = (
            pm.pivot(index="y", columns="x", values="pp")
            .reindex(index=pop_y, columns=pop_x, fill_value=0.0)
            .fillna(0.0)
        )
        X2d, Y2d = np.meshgrid(pop_x, pop_y, indexing="xy")
        point2d = np.array([X2d, Y2d, np.zeros_like(X2d)]).reshape(3, -1).T
        population = grid.to_numpy().reshape(-1)
    else:
        min_lon, min_lat = transformer_ll.transform(min_x, min_y)
        max_lon, max_lat = transformer_ll.transform(max_x, max_y)
        pm = pop_map.query("@min_lon<lon<@max_lon and @min_lat<lat<@max_lat").copy()
        pop_lon = np.sort(pm.lon.unique())
        pop_lat = np.sort(pm.lat.unique())
        pop_y = pop_lat
        grid = (
            pm.pivot(index="lat", columns="lon", values="pp")
            .reindex(index=pop_lat, columns=pop_lon, fill_value=0.0)
            .fillna(0.0)
        )
        Lon2d, Lat2d = np.meshgrid(pop_lon, pop_lat, indexing="xy")
        X2d, Y2d = transformer_xy.transform(Lon2d, Lat2d)
        point2d = np.array([X2d, Y2d, np.zeros_like(X2d)]).reshape(3, -1).T
        population = grid.to_numpy().reshape(-1)
    # Sanity: population vector must align with point2d columns
    if population.shape[0] != point2d.shape[0]:
        raise ValueError(
            f"Population/point grid mismatch: population={population.shape[0]} vs point2d={point2d.shape[0]}"
        )

    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 45000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    for i in range(len(flight)):
        thr = flight.thrust.values[i]
        delta = aero.pressure(flight.h.values[i]) / aero.pressure(0)
        cnt_per_eng = thr / delta / 2
        # ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        ns = interp_npd1(np.array([cnt_per_eng] * len(dist_0[i])), dist_0[i])
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    affected_pop = np.zeros(dist_0.shape)
    for i in range(len(flight)):
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    mask = affected_pop != 0
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)
    pp_aff_uni = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(len(flight), len(pop_y), -1), axis=0)
    pp_aff_all = sum(sum(agg_pop))
    return pp_aff_all, pp_aff_uni, flight


def cost_grid_cost(df_cost, flight):
    interpolant = top.tools.interpolant_from_dataframe(df_cost)
    cost = interpolant(
        np.array([flight.longitude.values, flight.latitude.values, flight.h.values])
    )
    flight = flight.assign(cost_grid=cost.full()[0])
    return flight


def get_npd_interpolator(ac):
    if ac.lower() != ac:
        print("For now A320 only")
        return
    npd = (
        pd.read_csv(curr_path + "/../data/npds_airbus.csv")
        .query("engine_name=='CFM56-5-A1'")
        .drop(columns=["engine_name"])
        .reset_index(drop=True)
    )
    thrusts = npd.thrust_n.to_numpy()
    dists = (
        np.array(npd.rename(columns=lambda x: x[:-3]).columns[:-1].astype(int))
        * aero.ft
    )
    noise_levels = npd.iloc[:, :-1].values

    interp_npd = RegularGridInterpolator(
        (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
    )
    return interp_npd


# %%
