# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pyproj import Transformer, CRS
from openap import aero, nav
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from openap import aero, nav, top, prop
from tqdm import tqdm
from traffic.data import navaids, airports

map_type = "D"
treshold = 60  # db
crs_3035 = CRS.from_epsg(3035)
crs_4326 = CRS.from_epsg(4326)
transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)
eham = nav.airport("EHAM")

flights = pd.read_csv(f"data_generated/optimal_flights_{map_type}.csv").query(
    "obj=='pop'"
)
flights0 = pd.read_csv(f"data_generated/optimal_flights_{map_type}.csv").query(
    "obj=='fuel'"
)
df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
df_real = pd.read_parquet(f"data_generated/opensky2024_centroids_{map_type}.parquet")

max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
min_x, min_y = transformer_xy.transform(*min_lola)
max_x, max_y = transformer_xy.transform(*max_lola)

pop_map = pd.read_parquet(f"data_raw/{map_type}032011_1K_cropped.parquet").rename(
    columns={"popul": "pp"}
)

pop_map = pop_map.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
pop_x = pop_map.x.unique()
pop_y = pop_map.y.unique()
X2d, Y2d, Z2d = np.meshgrid(pop_x, pop_y, [0])
point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T
npd = (
    pd.read_csv("data_generated/npds_new_airbus.csv")
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
# %%
pp_dict = []
for fid in tqdm(flights.fid.unique()):
    flight = flights.query(f"fid=='{fid}'")
    flight = flight.reset_index(drop=True)

    xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
    point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 45000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    for i in range(40):
        thr = flight.thrust.values[i]
        ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    aggregated_noise = np.sum(noise.reshape(40, len(pop_y), -1), axis=0)
    # plt.contour(pop_x,pop_lat,aggregated_noise.T, cmap = "Reds")
    # plt.plot(flight.longitude,flight.latitude)
    population = pop_map.pp.values.reshape(len(pop_x), len(pop_y))
    affected_pop = np.zeros(dist_0.shape)
    for i in range(40):
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    mask = affected_pop != 0

    # Get a single representative layer with unique values at each position
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)

    # Sum the unique values
    result = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_y), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append(
        {
            "fid": fid,
            "fuel": (flight.mass.values[0] - flight.mass.values[-1]),
            "cost_grid": flight.cost_grid.sum(),
            "people_affected_sum_all": people_affected,
            "pp_unique_sum": result,
        }
    )


df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(
    f"data_generated/pp_affected_{treshold}/pp_affected_noise_opt_{map_type}.csv",
    index=False,
)

# %%
pp_dict = []
for fid in tqdm(flights0.fid.unique()):
    flight = flights0.query(f"fid=='{fid}'")
    flight = flight.reset_index(drop=True)

    xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
    point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 45000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    for i in range(40):
        thr = flight.thrust.values[i]
        ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    aggregated_noise = np.sum(noise.reshape(40, len(pop_y), -1), axis=0)
    # plt.contour(pop_x,pop_lat,aggregated_noise.T, cmap = "Reds")
    # plt.plot(flight.longitude,flight.latitude)
    population = pop_map.pp.values.reshape(len(pop_x), len(pop_y))
    affected_pop = np.zeros(dist_0.shape)
    for i in range(40):
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    mask = affected_pop != 0

    # Get a single representative layer with unique values at each position
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)

    # Sum the unique values
    result = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_y), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append(
        {
            "fid": fid,
            "fuel": (flight.mass.values[0] - flight.mass.values[-1]),
            "cost_grid": flight.cost_grid.sum(),
            "people_affected_sum_all": people_affected,
            "pp_unique_sum": result,
        }
    )

df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(
    f"data_generated/pp_affected_{treshold}/pp_affected_fuel_opt_{map_type}.csv",
    index=False,
)
# %%
pp_dict = []
for fid in tqdm(df_real.flight_id.unique()):
    flight = df_real.query(f"flight_id=='{fid}'")
    flight = flight.reset_index(drop=True)

    xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
    point3d = np.array([xf, yf, flight.altitude.values * aero.ft]).reshape(3, -1).T
    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 45000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    for i in range(len(flight)):
        thr = flight.thrust.values[i]
        ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    aggregated_noise = np.sum(noise.reshape(len(flight), len(pop_y), -1), axis=0)
    # plt.contour(pop_x,pop_lat,aggregated_noise.T, cmap = "Reds")
    # plt.plot(flight.longitude,flight.latitude)
    population = pop_map.pp.values.reshape(len(pop_x), len(pop_y))
    affected_pop = np.zeros(dist_0.shape)
    for i in range(40):
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    mask = affected_pop != 0

    # Get a single representative layer with unique values at each position
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)

    # Sum the unique values
    result = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_y), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append(
        {
            "fid": fid,
            "fuel": (flight.mass.values[0] - flight.mass.values[-1]),
            # "cost_grid": flight.cost_grid.sum(),
            "people_affected_sum_all": people_affected,
            "pp_unique_sum": result,
        }
    )

df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(
    f"data_generated/pp_affected_{treshold}/pp_affected_real_{map_type}.csv",
    index=False,
)

# %%
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE

proj = ccrs.TransverseMercator(
    central_longitude=eham["lon"], central_latitude=eham["lat"]
)

trans = ccrs.PlateCarree()
# trans = CRS.from_epsg(3035)
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)
# ax.set_extent([3730500.0,4140500,3044500.0,3371500.0])

ax.add_feature(BORDERS, linestyle="dotted", alpha=0.2)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.2)
ax.set_title("day/night - " + map_type)
# ax.plot(f_real.longitude, f_real.latitude, transform=trans, c="cyan")

ax.scatter(
    pop_map.query("10<pp").x,
    pop_map.query("10<pp").y,
    c=pop_map.query("10<pp").pp,
    cmap="Reds",
    s=0.5,
    # transform=trans,
    # levels=15,
    norm=plt.Normalize(vmin=10, vmax=2000, clip=True),
    alpha=0.2,
)

norm = plt.Normalize(vmin=50, vmax=100, clip=True)
ax.contour(
    pop_x,
    pop_y,
    aggregated_noise,
    norm=norm,
    cmap="binary",
    # transform=trans,
    alpha=0.6,
)
ax.plot(
    xf,
    yf,
    lw=3.5,
    c="w",
    # transform=trans,
    alpha=0.5,
)
ax.plot(
    xf,
    yf,
    lw=3,
    c="k",
    # transform=trans,
    alpha=0.5,
)
ax.plot(
    xf,
    yf,
    lw=2,
    c="tab:blue",
    # transform=trans,
    alpha=0.5,
)
norm = plt.Normalize(vmin=0, vmax=10, clip=True)
# ax.contour(
#     pop_x,
#     pop_y,
#     agg_pop,
#     norm=norm,
#     # transform=trans,
#     levels=5,
#     cmap="viridis",
#     alpha=0.1
# )
norm = plt.Normalize(vmin=0, vmax=10, clip=True)
ax.scatter(
    X2d,
    Y2d,
    c=agg_pop,
    s=0.5,
    norm=norm,
    # transform=trans,
    # levels=5,
    alpha=0.1,
    cmap="viridis",
)
# ax.gridlines()

plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cmap = plt.cm.viridis.copy()
flight = pd.read_csv(f"data_generated/flights_noise_opt_{map_type}.csv").query(
    "fid==fid.iloc[0]"
)
Z = noise.reshape(40, len(pop_y), -1)
Z2 = affected_pop.reshape(40, len(pop_y), -1)
# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
zf = flight.altitude.values * aero.ft
fsc = flight.query("cost_grid==0")
zsc = fsc.altitude.values * aero.ft
xfsc, yfsc = transformer_xy.transform(fsc.longitude, fsc.latitude)
# Plot multiple layers of 2D contour plots
z_values = range(0, 40, 4)
# z_ = np.linspace(0, 36000*aero.ft, 40)  # Different z-values for each layer
for z in z_values:
    if z == 0:
        cntr = ax.contour(
            pop_x,
            pop_y,
            Z[z, :, :],
            200,
            zdir="z",
            offset=0,  # zf[z],
            cmap=cmap,
            alpha=1,
            norm=plt.Normalize(vmin=0, vmax=90, clip=True),
        )
    else:
        ax.contour(
            pop_x,
            pop_y,
            Z[z, :, :],
            200,
            zdir="z",
            offset=0,  # zf[z]/1000000,
            cmap=cmap,
            alpha=0.2,
            norm=plt.Normalize(vmin=0, vmax=90, clip=True),
        )

Z2 = Z
Z2 = np.zeros(Z.shape)
for z in z_values:
    ax.contourf(
        pop_x,
        pop_y,
        Z2[z, :, :],
        50,
        zdir="z",
        offset=zf[z],
        cmap=cmap,
        alpha=0.1,
        norm=plt.Normalize(vmin=0, vmax=90, clip=True),
    )


cbar = plt.colorbar(cntr, shrink=0.4, orientation="horizontal", pad=0.08)
# cbar.set_ticks(np.arange(0.05, 102.05, 20))
cbar.set_ticks(np.arange(0, 90.0, 20))
# cbar.set_ticklabels(["Low", "High"])

cbar.ax.set_xlabel("Noise, dBA", rotation=0, labelpad=5)
yticks = ax.get_yticks()
xticks = ax.get_xticks()


# ax.set_xticks([xticks[1],xticks[-2]])
# ax.set_yticks([yticks[1],yticks[-2]])
# ll_min = transformer_ll.transform(xticks[1],yticks[1])
# ll_max = transformer_ll.transform(xticks[-2],yticks[-2])
# ax.set_xticklabels([np.round(ll_min[0],2),np.round(ll_max[0],2)])
# ax.set_yticklabels([np.round(ll_min[1],2),np.round(ll_max[1],2)])
ax.set_xticks([])
ax.set_yticks([])
ax.plot3D(xf, yf, zf, c="tab:red")
ax.plot3D([xfsc[0], xfsc[0]], [yfsc[0], yfsc[0]], [0, zsc[0]], c="navy", linestyle="--")
ax.scatter3D(xfsc[0], yfsc[0], zsc[0], c="navy", s=40, marker="x")
# ax.scatter3D(xfsc[0], yfsc[0], 0, c="k", s=100)

ax.contourf(
    pop_x,
    pop_y,
    Z2[0, :, :],
    levels=2,
    zdir="z",
    offset=zsc[0],
    cmap="cividis",
    alpha=0.2,
    # norm=plt.Normalize(vmin=0, vmax=90, clip=True),
)
zticks = ax.get_zticks()
# ax.scatter3D(xf[0], yf[0], zf[0], c="g", s=100)

ax.set_zticklabels((np.round(zticks / aero.ft / 1000, 0)).astype(int))
# ax.set_zlim(0, 10000)
# ax.scatter3D(
#         pop_x,
#         pop_y,
#         0,
#         # c = pop_map.pp,
#         # 200,
#         # zdir="z",
#         # offset=z*1000,
#         cmap="Reds",
#         alpha=0.3,
#         norm=plt.Normalize(vmin=0, vmax=5000, clip=True),
#     )
# # Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Altitude, 1000 ft")
# ax.set_title("3D Contour Plot with Layers")
plt.savefig(f"figs/3dplot.png", pad_inches=0.3, bbox_inches="tight", dpi=200)
plt.show()
# Show the plot
plt.show()

# %%
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE

from matplotlib.colors import LinearSegmentedColormap

ncolors = 256
color_array = plt.get_cmap("viridis")(range(ncolors))

# change alpha values
color_array[:, -1] = np.linspace(1.0, 0.0, ncolors)
# color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(
    name="viridis_alpha_r", colors=color_array
)
plt.colormaps.register(cmap=map_object)


# %%
# proj = ccrs.PlateCarree()
# trans = ccrs.PlateCarree()
# # trans = CRS.from_epsg(3035)
# fig, ax = plt.subplots(
#     1,
#     1,
#     figsize=(6, 6),
#     subplot_kw=dict(projection=proj),
# )
# ax.set_extent([4, 5.5, 51.1, 52.65])

# ax.add_feature(BORDERS, linestyle="dotted", alpha=0.6)
# ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.6)
# # ax.set_title("day/night - " + map_type)
# # ax.plot(df_real.longitude, df_real.latitude, transform=trans, c="cyan")
# ax.plot(
#     [flonmax, flonmax, flonmin, flonmin, flonmax],
#     [flatmin, flatmax, flatmax, flatmin, flatmin],
#     transform=trans,
#     c="tab:red",
#     linestyle="dashed",
# )
# ax.scatter(
#     pop_map0.query("10<pp").lon,
#     pop_map0.query("10<pp").lat,
#     c=pop_map0.query("10<pp").pp,
#     cmap="Reds",
#     s=3,
#     transform=trans,
#     # levels=15,
#     norm=plt.Normalize(vmin=10, vmax=2000, clip=True),
#     alpha=0.3,
# )

# norm = plt.Normalize(vmin=0, vmax=1, clip=True)
# pop_lon, pop_lat = transformer_ll.transform(X2d, Y2d)
# noise50 = np.sum(noise1 > 50, axis=0)
# noise60 = np.sum(noise1 > 60, axis=0)
# noise75 = np.sum(noise1 > 75, axis=0)
# cmap = ["Purples_r", "Wistia_r", "Greens_r"]
# for i, noise_n in enumerate([noise50, noise60, noise75]):
#     ax.contour(
#         pop_lon[:, :, 0],
#         pop_lat[:, :, 0],
#         noise_n,
#         norm=norm,
#         cmap=cmap[i],
#         levels=0,
#         transform=trans,
#         alpha=0.7,
#     )
# lonf, latf = transformer_ll.transform(xf, yf)
# ax.plot(
#     lonf,
#     latf,
#     lw=3.5,
#     c="w",
#     transform=trans,
#     alpha=0.5,
# )
# ax.plot(
#     lonf,
#     latf,
#     lw=3,
#     c="k",
#     transform=trans,
#     alpha=0.5,
# )
# ax.plot(
#     lonf,
#     latf,
#     lw=2,
#     c="k",
#     transform=trans,
#     alpha=0.5,
# )
# # norm = plt.Normalize(vmin=0, vmax=100, clip=True)
# # cntr = ax.contourf(
# #     pop_lon[:, :, 0],
# #     pop_lat[:, :, 0],
# #     agg_pop,
# #     norm=norm,
# #     transform=trans,
# #     levels=np.linspace(10, 10000, 130),
# #     cmap="viridis",
# #     alpha=1,
# # )
# norm = plt.Normalize(vmin=-10, vmax=100, clip=True)
# # ax.scatter(
# #     pop_lon,
# #     pop_lat,
# #     c=agg_pop,
# #     s=0.1,
# #     norm=norm,
# #     transform=trans,
# #     # levels=5,
# #     alpha=0.2,
# #     cmap="viridis",
# # )
# # ax.gridlines()
# # ax.scatter(
# #     pop_map.query("10<pp").lon,
# #     pop_map.query("10<pp").lat,
# #     c=pop_map.query("10<pp").pp,
# #     cmap="Reds",
# #     s=0.1,
# #     transform=trans,
# #     # levels=15,z
# #     norm=plt.Normalize(vmin=10, vmax=10000, clip=True),
# #     alpha=0.2,
# # )
# # cbar = plt.colorbar(cntr)
# plt.savefig(
#     f"figs/contours_n_{map_type}.png", pad_inches=0, bbox_inches="tight", dpi=500
# )
# plt.show()

# %%
