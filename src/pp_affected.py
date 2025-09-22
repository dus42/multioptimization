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
import os
curr_path = os.path.dirname(os.path.realpath(__file__))
#%%


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
    fuelflow = FuelFlow(ac)
    thrust = Thrust(ac)
    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
    transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)
    thr = flight.thrust
    ff = fuelflow.enroute(flight.mass.values[0], flight.tas.values,flight.altitude.values, flight.vertical_rate.values)
    flight = flight.assign(thrust = thr, fuel = ff*flight.ts.diff()).fillna(0)
    interp_npd = get_npd_interpolator(ac)
    if grid_type == "xy":
        xf,yf = transformer_xy.transform(flight.longitude, flight.latitude)
        point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
        min_x, min_y = (xf.min()-8000, yf.min()-8000)
        max_x, max_y = (xf.max()+8000, yf.max()+8000)
        pop_map = pop_map.query("@min_x<x<@max_x and @min_y<y<@max_y")
        pop_x = pop_map.x.unique()
        pop_y = pop_map.y.unique()
        X2d, Y2d, Z2d = np.meshgrid(pop_x, pop_y, [0])
        point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T
    else: 
        xf,yf = transformer_xy.transform(flight.longitude, flight.latitude)
        point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
        min_x, min_y = (xf.min()-8000, yf.min()-8000)
        max_x, max_y = (xf.max()+8000, yf.max()+8000)
        min_lon,min_lat = transformer_ll.transform(min_x,min_y)
        max_lon,max_lat = transformer_ll.transform(max_x,max_y)
        pop_map = pop_map.query("@min_lon<lon<@max_lon and @min_lat<lat<@max_lat")
        pop_lon = pop_map.lon.unique()
        pop_lat = pop_map.lat.unique()
        pop_y=pop_lat
        Lon2d, Lat2d, Z2d = np.meshgrid(pop_lon, pop_lat, [0])
        X2d, Y2d = transformer_xy.transform(Lon2d, Lat2d)
        point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T

    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 45000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    for i in range(len(flight)):
        thr = flight.thrust.values[i]
        ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    population = pop_map.pp.values
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
    if ac.lower() !=ac:
        print("For now A320 only")
        return
    npd = (
    pd.read_csv(curr_path+"/../data_generated/npds_new_airbus.csv")
    .query("engine_name=='CFM56-5-A1'")
    .drop(columns=["engine_name"])
    .reset_index(drop=True)
    )
    thrusts = npd.thrust_n.to_numpy()
    dists = (
        np.array(npd.rename(columns=lambda x: x[:-3]).columns[:-1].astype(int)) * aero.ft
    )
    noise_levels = npd.iloc[:, :-1].values

    interp_npd = RegularGridInterpolator(
        (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
    )
    return interp_npd

# %%
