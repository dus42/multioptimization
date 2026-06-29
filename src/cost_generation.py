# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pyproj import Transformer, CRS
from openap import aero, nav
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE

from tqdm import tqdm

# %%


def cost_generator(
    pop_map,
    grid_type,
    airport,
    alt_max=25000,
    nodes=(30, 20, 20),
    bounds=None,
    plot=False,
):
    """
    Generation of a cost grid using population map.
    pop_map: pd.DataFrame, the population map with columns ["x", "y", "pp"] and/or ["lon", "lat", "pp"]
    grid_type: "xy" or "ll", "xy" - grid based on x and y coordinates, "ll" - grid based on longitude and latitude
    airport: str, the ICAO code of the airport
    bounds: list of 4 floats, [min_x, min_y, max_x, max_y] or [min_lon, min_lat, max_lon, max_lat]
    nodes: tuple of 3 ints, the number of nodes in x, y and z directions
    alt_max: int, the maximum altitude of the cost grid in feet, default = 25000ft
    plot: bool, if True, the plot of the cost grid and poplation map is shown, Default = False
    return: pd.DataFrame with columns ["longitude", "latitude", "altitude", "cost"], the cost grid

    """

    nx, ny, nz = nodes
    airport = nav.airport(airport)

    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
    transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)

    start = (airport["lat"], airport["lon"])
    start_xy = transformer_xy.transform(start[1], start[0])
    if bounds is not None:
        if np.max(bounds) > 180:
            min_x, min_y, max_x, max_y = bounds
            min_lon, min_lat = transformer_ll.transform(bounds[0], bounds[1])
            max_lon, max_lat = transformer_ll.transform(bounds[2], bounds[3])
        else:
            min_lon, min_lat, max_lon, max_lat = bounds
            min_x, min_y = transformer_xy.transform(bounds[0], bounds[1])
            max_x, max_y = transformer_xy.transform(bounds[2], bounds[3])
    else:
        min_x, min_y = (
            start_xy[0] - 300000,
            start_xy[1] - 300000,
        )  # 300 km from the airport
        max_x, max_y = (
            start_xy[0] + 300000,
            start_xy[1] + 300000,
        )  # 300 km from the airport
        min_lon, min_lat = transformer_ll.transform(min_x, min_y)
        max_lon, max_lat = transformer_ll.transform(max_x, max_y)

    min_x, min_y = transformer_xy.transform(min_lon, min_lat)
    max_x, max_y = transformer_xy.transform(max_lon, max_lat)

    lonp = np.linspace(min_lon, max_lon, nx)
    latp = np.linspace(min_lat, max_lat, ny)
    # yp = np.linspace(min_y, max_y, ny)
    # xp = np.linspace(min_x, max_x, nx)
    altp = np.linspace(0, alt_max + 2000, nz)

    Lon_cost, Lat_cost, Alt_cost = np.meshgrid(lonp, latp, altp, indexing="ij")
    X_cost, Y_cost = transformer_xy.transform(Lon_cost, Lat_cost)
    Z_cost = Alt_cost * aero.ft
    if grid_type == "ll":
        pop_map = pop_map.query(
            "@min_lon<lon<@max_lon and @min_lat<lat<@max_lat"
        ).fillna(0)
        x, y = transformer_xy.transform(pop_map.lon, pop_map.lat)
        pop_map = pop_map.assign(x=x, y=y)
        pop = pop_map.query("pp > 0")[["x", "y", "pp"]].to_numpy()
        xp, yp, pp = pop[:, 0], pop[:, 1], pop[:, 2]
        point2d = np.column_stack([xp, yp, np.zeros_like(xp)])
    else:
        pop_map = pop_map.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
        pop = pop_map.query("pp > 0")[["x", "y", "pp"]].to_numpy()
        xp, yp, pp = pop[:, 0], pop[:, 1], pop[:, 2]
        point2d = np.column_stack([xp, yp, np.zeros_like(xp)])
    cost = np.zeros((nx, ny, nz), dtype=np.float64)
    eps2 = 1.0
    for x in tqdm(range(nx)):
        for y in range(ny):
            p3 = np.column_stack(
                [X_cost[x, y, :], Y_cost[x, y, :], Z_cost[x, y, :]]
            )  # (nz, 3)
            dist = distance_matrix(p3, point2d)  # (nz, Npop)
            w = 1.0 / (dist * dist + eps2)
            cost[x, y, :] = (w * pp[None, :]).sum(axis=1)

    df_cost = (
        (
            pd.DataFrame(
                np.array([Lon_cost, Lat_cost, Alt_cost, cost]).reshape(4, -1).T,
                columns=["longitude", "latitude", "altitude", "cost"],
            )
            .assign(height=lambda x: x.altitude * 0.3048)
            .assign(cost=lambda x: np.where(x.altitude > alt_max, 0, x.cost))
        )
        .sort_values(by=["longitude", "latitude", "height"])
        .fillna(0)
    )

    if plot:
        # %%
        proj = ccrs.TransverseMercator(
            central_longitude=start[1], central_latitude=start[0]
        )
        # proj = ccrs.PlateCarree()
        trans = ccrs.PlateCarree()

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(6, 6),
            subplot_kw=dict(projection=proj),
        )
        ax.set_extent(
            [
                min_lon - 0.1,
                max_lon + 0.1,
                min_lat - 0.1,
                max_lat + 0.1,
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(BORDERS, linestyle="solid", alpha=1)
        ax.add_feature(COASTLINE, linestyle="solid", alpha=1)

        norm = plt.Normalize(vmin=0.000104, vmax=0.0003)
        ax.scatter(
            pop_map.query(f"20000>pp >= 100").x,
            pop_map.query(f"20000>pp >= 100").y,
            c=pop_map.query(f"20000>pp >= 100").pp,
            s=1,
            alpha=1,
            # transform=trans,
            transform=ccrs.epsg(3035),
            cmap="Reds",
            norm=plt.Normalize(vmin=100, vmax=10000),
        )
        ax.contour(
            df_cost.longitude.to_numpy().reshape(nx, ny, nz)[:, :, 4],
            df_cost.latitude.to_numpy().reshape(nx, ny, nz)[:, :, 4],
            df_cost.cost.to_numpy().reshape(nx, ny, nz)[:, :, 4],
            levels=40,
            alpha=0.5,
            norm=norm,
            transform=trans,
            cmap="cool",
        )
        ax.scatter(
            df_cost.longitude.to_numpy().reshape(nx, ny, nz)[:, :, 4],
            df_cost.latitude.to_numpy().reshape(nx, ny, nz)[:, :, 4],
            s=5,
            c=df_cost.cost.to_numpy().reshape(nx, ny, nz)[:, :, 4],
            # levels=40,
            alpha=1,
            # norm=norm,
            transform=trans,
            cmap="cool",
        )

        # ax.scatter(
        #     # transformer_ll.transform(
        #     #     pop_map.query(f"20000>pp >= 10").x,
        #     #     pop_map.query(f"20000>pp >= 10").y,
        #     # )[0],
        #     # transformer_ll.transform(
        #     #     pop_map.query(f"20000>pp >= 10").x,
        #     #     pop_map.query(f"20000>pp >= 10").y,
        #     # )[1],
        #     pop_map.query(f"20000>pp >= 10").lon,
        #     pop_map.query(f"20000>pp >= 10").lat,
        #     c=pop_map.query(f"20000>pp >= 10").pp,
        #     s=1,
        #     alpha=1,
        #     transform=trans,
        #     cmap="viridis",
        #     norm=plt.Normalize(vmin=100, vmax=10000),
        # )
        ax.gridlines(color="k", draw_labels=True, alpha=0.1)

        plt.tight_layout()
        plt.savefig(f"cost_map_2023.png", bbox_inches="tight", dpi=300)
        # plt.show()
        # %%
    return df_cost


# if map == "D01" or map == "N01":
#     pop_map = pd.read_parquet(f"data_raw/{map}2011_1K_cropped.parquet").rename(columns = {"popul":"pp"})
#     grid_type = "xy"
#     bounds = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
#     red_pop_map, point2d = pop_map_preprocessing(pop_map, "xy", [min_xy[0], min_xy[1], max_xy[0], max_xy[1]])

# elif map == "st":
#     pop_map = pd.read_parquet(f"data_raw/pop_static.parquet")
#     grid_type = "ll"
#     bounds = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
#     red_pop_map, point2d = pop_map_preprocessing(pop_map, "ll", [min_xy[0], min_xy[1], max_xy[0], max_xy[1]],max_len=230)
# norm = plt.Normalize(vmin=0.0, vmax=0.02)
# print(f"map:{map}")
# print(f"sum_cost:{df_cost.cost.sum()}")
# print(f"sum_pp_red:{round(red_pop_map.pp.sum()/1e6,4)} million people")
# print(f"sum_pp_orig:{round(pop_map.pp.sum()/1e6,4)} million people")
# print(f"max:{df_cost.cost.max()}")
# print(f"50:{np.percentile(df_cost.cost,50)}")
# plt.scatter(
#     df_cost.query("cost>0").longitude,
#     df_cost.query("cost>0").latitude,
#     c=df_cost.query("cost>0").cost,
#     norm=norm,
#     s=100,
# )

# %%
