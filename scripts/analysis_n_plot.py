# %%
import numpy as np
import pandas as pd
from pyproj import Transformer, CRS
from openap import aero, nav
from scipy.interpolate import RegularGridInterpolator
from openap import aero, nav, top, prop
from tqdm import tqdm
from traffic.data import navaids, airports
from traffic.core import Flight, Traffic
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import sys
import os
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.ticker as mticker

sys.path.append(os.path.abspath("../src/"))
from pp_affected import pp_affected

# %%
### Plotting 2 different altitude layers of df_cost
map_type = "DN"
eham = nav.airport("EHAM")
nx, ny, nz = 50, 45, 23
df_cost = pd.read_csv(f"../data_generated/df_cost_{map_type}.csv")
if map_type == "D" or map_type == "N":
    pop_map = (
        pd.read_parquet(f"../data_raw/{map_type}032011_1K_cropped.parquet")
        .rename(columns={"popul": "pp"})
        .fillna(0)
    )
else:
    pop_map = pd.read_parquet("../data_raw/pop_static.parquet")
    df_cost["cost"] = df_cost["cost"] / 2

proj = ccrs.TransverseMercator(
    central_longitude=eham["lon"], central_latitude=eham["lat"]
)
trans = ccrs.PlateCarree()
altp = np.linspace(0, 22000, 20)


fig = plt.figure(figsize=(7, 4))
gs = matplotlib.gridspec.GridSpec(1, 9)
ax0 = fig.add_subplot(gs[0:1, 0:4], projection=proj)
ax1 = fig.add_subplot(gs[0:1, 4:9], projection=proj)
ialt1 = 2
ialt2 = 11
norm = plt.Normalize(vmin=0.0001, vmax=0.09, clip=True)
cost_contour = ax0.contourf(
    df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.cost.values.reshape(nx, ny, nz)[:, :, ialt1],
    levels=15,
    alpha=1,
    norm=norm,
    transform=trans,
    cmap="managua",
    label="grid_cost",
)
cost_contour2 = ax1.contourf(
    df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.cost.values.reshape(nx, ny, nz)[:, :, ialt2],
    levels=15,
    alpha=1,
    norm=norm,
    transform=trans,
    cmap="managua",
    label="grid_cost",
)
ax0.set_title(f"a) Altitude: {int(round(altp[ialt1], -2))} ft", fontsize=11)
ax1.set_title(f"b) Altitude: {int(round(altp[ialt2], -2))} ft", fontsize=11)


for ax in [ax0, ax1]:
    ax.set_extent([3.4, 6.8, 50.8, 53.4])
    ax.add_feature(BORDERS, color="k", alpha=0.3)
    ax.add_feature(COASTLINE, color="k", alpha=0.3)
    norm2 = plt.Normalize(vmin=100, vmax=8000)

    pop_vis = ax.scatter(
        pop_map.query("pp>1000").lon.values,
        pop_map.query("pp>1000").lat.values,
        # c=pop_map.query("pp>100").pp.values,
        c="k",
        s=0.25,
        transform=trans,
        norm=norm2,
        alpha=0.11,
        # cmap="binary",
    )

    gl = ax.gridlines(draw_labels=False, color="gray", alpha=0.5, ls="--")
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlocator = mticker.FixedLocator([4, 5, 6])
    gl.ylocator = mticker.FixedLocator([51, 52, 53])
    if ax == ax1:
        gl.left_labels = False
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}


cbar2 = plt.colorbar(
    cost_contour,
    ax=ax1,
    orientation="vertical",
    shrink=0.6,
    aspect=12,
    pad=0.06,
)
cbar2.set_ticks([0, 0.104])
cbar2.set_ticklabels(["Low", "High"])

cbar2.ax.set_xlabel("Grid\ncost", rotation=0, labelpad=-185, fontsize=11)

plt.tight_layout()
plt.savefig("../figs/cost_vs_pop.png", bbox_inches="tight", dpi=300)
plt.show()
# %%
## Plot clusters and centoirds
cmap = plt.get_cmap("turbo")
colors = cmap(np.linspace(0, 1, 13))  # Assign unique colors
map_type = "DN"
df_all = pd.read_parquet(
    f"../data_generated/opensky2024_clustered_flights_{map_type}.parquet"
)
df_centroids = pd.read_parquet(
    f"../data_generated/opensky2024_centroids_{map_type}.parquet"
)
if map_type == "D" or map_type == "N":
    pop_map = (
        pd.read_parquet(f"../data_raw/{map_type}012011_1K_cropped.parquet")
        .rename(columns={"popul": "pp"})
        .fillna(0)
    )
else:
    pop_map = pd.read_parquet("../data_raw/pop_static.parquet")
# proj = ccrs.PlateCarree()
proj = ccrs.TransverseMercator(
    central_longitude=eham["lon"], central_latitude=eham["lat"]
)
trans = ccrs.PlateCarree()

fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 4),
    subplot_kw=dict(projection=proj),
)
ax.set_extent([3, 6.8, 50.8, 53.4])
ax.add_feature(BORDERS, color="k", alpha=0.3)
ax.add_feature(COASTLINE, color="k", alpha=0.3)

# norm2 = plt.Normalize(vmin=10, vmax=10000)

pop_vis = ax.scatter(
    pop_map.query("pp>1000").lon.values,
    pop_map.query("pp>1000").lat.values,
    # c=pop_map.query("pp>100").pp.values,
    c="k",
    s=0.15,
    transform=trans,
    norm=norm2,
    alpha=0.21,
    # cmap="binary",
)
gl = ax.gridlines(draw_labels=False, color="gray", alpha=0.5, ls="--")
gl.bottom_labels = True
gl.left_labels = True
gl.xlocator = mticker.FixedLocator([4, 5, 6])
gl.ylocator = mticker.FixedLocator([51, 52, 53])
gl.xlabel_style = {
    "size": 6,
}
gl.ylabel_style = {
    "size": 6,
}
# ax.set_facecolor("lightgrey")
for fid in df_all.flight_id.unique()[:]:
    flight = df_all.query("flight_id==@fid")
    ax.plot(
        flight.longitude,
        flight.latitude,
        color=colors[flight.cluster.values[0]],
        lw=2,
        linestyle="dashed",
        alpha=0.2,
        transform=trans,
    )
for fid in sorted(df_centroids.cluster.unique()):
    flight = df_centroids.query("cluster==@fid")
    ax.plot(
        flight.longitude,
        flight.latitude,
        color="w",
        lw=3.5,
        transform=trans,
        alpha=0.8,
    )
    ax.plot(
        flight.longitude,
        flight.latitude,
        color="k",
        lw=3.2,
        transform=trans,
        alpha=0.8,
    )
    ax.plot(
        flight.longitude,
        flight.latitude,
        color="w",
        lw=2.9,
        transform=trans,
        alpha=0.8,
    )
    ax.plot(
        flight.longitude,
        flight.latitude,
        color=colors[flight.cluster.values[0]],
        lw=2.6,
        transform=trans,
        alpha=0.8,
        label=str(flight.cluster.values[0]),
    )
plt.legend(loc="lower center", ncols=7, fontsize=6)
plt.tight_layout()
plt.savefig("../figs/clusters_n_centroids.png", bbox_inches="tight", dpi=200)
plt.show()
# %%
## Fuel vs Noise
from openap.casadi import aero as aero_casadi
from openap import top

import casadi
from functools import partial

ac = "a320"
m0 = 0.8
eham = nav.airport("EHAM")
ebbr = nav.airport("EBBR")
start = (eham["lat"], eham["lon"])
# end = (navaids["KEKIX"].latitude + 0.3, navaids["KEKIX"].longitude + 0.75)
end = (ebbr["lat"], ebbr["lon"])
nodes = 30
h_end = 36_000 * aero.ft
flights = None
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
# df_cost = df_cost.assign(cost=df_cost.cost/2)
optimizer = top.Climb(ac, start, end, m0=m0)
optimizer.setup(nodes=nodes, max_iteration=5000)
flights = []
flight = optimizer.trajectory(
    objective="fuel",
    h_end=h_end,
)

flight = flight.assign(fid=f"id{4}", num=0)
flights.append(flight)


def obj_noise(x, u, dt, optimizer, number, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = np.arctan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return number * cost * thrust * dt + fuel


# for i,number in tqdm(enumerate([10,5,1, 0.004, 0.001, ])):
for i, number in tqdm(enumerate([0.0009, 0.0003, 0.00015, 0.00003])):
    optimizer = top.Climb(ac, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=15000, debug=False)
    objective = partial(obj_noise, optimizer=optimizer, number=number)
    interpolant = top.tools.interpolant_from_dataframe(df_cost)
    flight = (
        optimizer.trajectory(
            objective=objective,
            interpolant=interpolant,
            h_end=h_end,
        )
        .assign(fid=f"id{i}")
        .assign(num=number)
    )
    flights.append(flight)
flights = pd.concat(flights)
cost = interpolant(
    np.array([flights.longitude.values, flights.latitude.values, flights.h.values])
)
flights = flights.assign(cost_grid=cost.full()[0]).sort_values(by=["num", "ts"])

flights.to_csv("../data_generated/fligths_spectrum.csv", index=False)
# %%
import openap

max_fuel = True
# max_fuel = False
if max_fuel:
    flights = pd.read_csv("../data_generated/fligths_spectrum_max_fuel.csv")
else:
    flights = pd.read_csv("../data_generated/fligths_spectrum.csv")
drag = openap.Drag("a320", wave_drag=True)

D = drag.clean(flights.mass, flights.tas, flights.altitude, flights.vertical_rate)
gamma = np.arctan2(
    flights.vertical_rate * openap.aero.fpm, flights.tas * openap.aero.kts
)
thrust = D + flights.mass * 9.81 * np.sin(gamma)
flights = flights.assign(thrust=thrust)
# %%
max_fuel = True
if max_fuel:
    coef = "Max fuel"
else:
    coef = r"$c_{dn}$"
latex_tab = []
for i, num in enumerate(flights.num.unique()):
    flight = flights.query("num==@num")
    latex_tab.append(
        {
            "fid": flight.fid.iloc[0],
            coef: flight.num.iloc[0],
            "Fuel consumed, kg": flight.mass.iloc[0] - flight.mass.iloc[-1],
            r"$C_{grid}$": flight.cost_grid.sum(),
            r"$T \times C_{grid}$": (flight.cost_grid * flight.thrust).sum(),
        }
    )
latex_tab = pd.DataFrame().from_dict(latex_tab).sort_values(by=["fid"])
latex_tab["Fuel spent, kg"] = latex_tab["Fuel consumed, kg"].apply(lambda x: f"{x:.2f}")
# latex_tab[r"$C_{grid}$"]=latex_tab[r"$C_{grid}$"].apply(lambda x: f"{x:.2f}")
if max_fuel:
    latex_tab["Max fuel"] = latex_tab["Max fuel"].apply(lambda x: f"{x:.3f}")
latex_tab[r"$T \times C_{grid}$"] = (
    latex_tab[r"$T \times C_{grid}$"].astype(int).apply(lambda x: f"{x:,}")
)

print(latex_tab.drop(columns=["Fuel consumed, kg"]).to_latex(escape=False, index=False))
# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.cm as cm

nx, ny, nz = 50, 45, 23
flights = pd.read_csv("../data_generated/fligths_spectrum.csv")
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
eham = nav.airport("EHAM")
proj = ccrs.PlateCarree()  # TransverseMercator(
#     central_longitude=eham["lon"], central_latitude=eham["lat"]
# )
trans = ccrs.PlateCarree()
plot_extent = [
    min(flights.longitude.values) - 0.3 - 0.15,
    max(flights.longitude.values) + 0.3 + 0.15,
    min(flights.latitude.values) - 0.3 - 0.15,
    max(flights.latitude.values) + 0.3 + 0.15,
]
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)
ax.set_extent(plot_extent)
ax.add_feature(BORDERS, linestyle="dotted", alpha=1)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=1)


df_c = df_cost  # .assign(cost=lambda x: np.where(x.cost > 0.8, 0.8, x.cost)).assign(cost=lambda x: np.where(x.cost < 0.01, 0.01, x.cost))


norm = plt.Normalize(vmin=0.0001, vmax=0.06)
contr = ax.contourf(
    df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.cost.values.reshape(nx, ny, nz)[:, :, 6],
    levels=15,
    alpha=0.4,
    norm=norm,
    transform=trans,
    cmap="binary",
    # label = "population"
)
props = dict(boxstyle="round", facecolor="w")
ax.text(
    eham["lon"] + 0.06,
    eham["lat"] + 0.04,
    "EHAM",
    transform=trans,
    verticalalignment="top",
    bbox=props,
    alpha=0.8,
)


cmap = plt.get_cmap("viridis_r")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "narrow_viridis", cmap(np.linspace(0.3, 0.8, 256))
)

colors_v = cmap(np.linspace(0, 1, len(flights.num.unique())))  # Assign unique colors
norm_fid = matplotlib.colors.Normalize(vmin=0, vmax=5)  # Normalize for color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm_fid)  # For colorbar
sm.set_array([])
ax.scatter(
    eham["lon"],
    eham["lat"],
    c="k",
    s=20,
)
for i, num in enumerate(flights.num.unique()):
    flight = flights.query("num==@num")
    # flight0= flights0.query("num==@num")

    if i != 0:
        ax.plot(
            flight.longitude,
            flight.latitude,
            color=colors_v[(i)],
            lw=2,
            linestyle="dashed",
            transform=trans,
            label=f"Noise optimal{num}",
        )
        ax.plot(
            flight.query("cost_grid>0").longitude,
            flight.query("cost_grid>0").latitude,
            color=colors_v[(i)],
            lw=2,
            # linestyle="dashed",
            transform=trans,
            # label=f"Noise optimal{num}",
        )
    else:
        ax.plot(
            flight.longitude,
            flight.latitude,
            color=colors_v[(i)],
            lw=2,
            # linestyle="dashed",
            transform=trans,
            label=f"Noise optimal{num}",
        )


cbar = fig.colorbar(
    sm, ax=ax, orientation="horizontal", shrink=0.4, fraction=0.03, pad=0.02
)
cbar.set_ticks(np.linspace(0, 5, 5))
# cbar.set_ticks([0,5])
# cbar.set_ticklabels(["Fuel optimal", "Population optimal"])
cbar.set_ticklabels([0.0, 0.3, 1.5, 3.0, 9.0])
# cbar.set_label(r"Fuel optimal   →   Population optimal")
cbar.set_label(r"Cost-weighting factor $c_{dn}, × 10^{-4}$")
# ax.legend()


plt.tight_layout()
plt.savefig("../figs/noise_fuel_spectrum_flights.png", bbox_inches="tight")
plt.show()
# %%
# %% Plot vertical rate
fig, (ax0) = plt.subplots(1, 1, figsize=(6, 4))
for i, num in enumerate(flights.num.unique()):
    flight = flights.query("num==@num")
    ax0.plot(
        flight.ts,
        flight.vertical_rate,
        lw=1,
        # linestyle="dashed",
        color=colors_v[(i)],
        alpha=0.5,
        # transform=trans,
        label=f"Noise optimal{num}" if num != 0 else f"Fuel optimal",
    )
    ax0.plot(
        flight.query("cost_grid>0").ts,
        flight.query("cost_grid>0").vertical_rate,
        lw=2,
        color=colors_v[(i)],
    )
    ax0.scatter(
        flight.query("cost_grid>0").ts.iloc[-1],
        flight.query("cost_grid>0").vertical_rate.iloc[-1],
        marker="*",
        color=colors_v[(i)],
    )
    ax0.plot(
        [
            flight.query("cost_grid>0").ts.iloc[-1],
            flight.query("cost_grid>0").ts.iloc[-1],
        ],
        [flight.query("cost_grid>0").vertical_rate.iloc[-1], 0],
        lw=1,
        linestyle="dashed",
        color=colors_v[(i)],
    )
ax0.text(105, 3200, "Altitude where \ncost grid reaches 0", fontsize=8.7)
ax0.annotate(
    "", xytext=(275, 3600), xy=(310, 3950), arrowprops=dict(arrowstyle="->", color="r")
)
ax0.set_ylim(0, 4200)
plt.xlabel("Time, s", fontsize=9)
plt.ylabel("Vertical rate, ft/min", fontsize=9)

cbar = fig.colorbar(
    sm,
    ax=ax0,
    orientation="vertical",
    shrink=0.4,
    fraction=0.03,
)
cbar.set_ticks([0, 5])
cbar.set_ticklabels(["Fuel \noptimal", "Population \noptimal"], fontsize=8)
plt.show()
# %%
## Fuel vs Noise max fuel
from openap.casadi import aero as aero_casadi
from openap import top
import casadi
from functools import partial

ac = "a320"
m0 = 0.8
eham = nav.airport("EHAM")
ebbr = nav.airport("EBBR")
start = (eham["lat"], eham["lon"])
# end = (navaids["KEKIX"].latitude + 0.3, navaids["KEKIX"].longitude + 0.75)
end = (ebbr["lat"], ebbr["lon"])
nodes = 30
h_end = 36_000 * aero.ft
flights = None
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
# df_cost = df_cost.assign(cost=df_cost.cost / 2)
optimizer = top.Climb(ac, start, end, m0=m0)
optimizer.setup(nodes=nodes, max_iteration=5000)
flights = []
flight = optimizer.trajectory(
    objective="fuel",
    h_end=h_end,
)

flight_f = flight.assign(fid=f"id{4}", num=0)
flights.append(flight_f)


def obj_noise_max_fuel(x, u, dt, optimizer, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    v = aero_casadi.mach2tas(mach, h)
    D = optimizer.drag.clean(m, v / aero.kts, h / aero.ft, vs / aero.fpm)
    gamma = np.arctan2(vs, v)
    thrust = D + m * 9.81 * casadi.sin(gamma)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=False, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.001 * cost * thrust * dt + fuel


# for i,number in tqdm(enumerate([10,5,1, 0.004, 0.001, ])):
for i, number in tqdm(enumerate([1.02, 1.011, 1.005, 1.001])):
    optimizer = top.Climb(ac, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=45000, debug=False)
    objective = partial(obj_noise_max_fuel, optimizer=optimizer)
    interpolant = top.tools.interpolant_from_dataframe(df_cost)
    flight = optimizer.trajectory(
        objective=objective,
        interpolant=interpolant,
        h_end=h_end,
        max_fuel=(flight_f.mass.values[0] - flight_f.mass.values[-1]) * number,
    )
    if flight is None:
        print(number)
        continue
    flight = flight.assign(fid=f"id{i}").assign(num=number)
    flights.append(flight)
# %%
flights = pd.concat(flights)
cost = interpolant(
    np.array([flights.longitude.values, flights.latitude.values, flights.h.values])
)
flights = flights.assign(cost_grid=cost.full()[0]).sort_values(by=["num", "ts"])
flights.to_csv("../data_generated/fligths_spectrum_max_fuel.csv", index=False)
print(flights)
# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.cm as cm

nx, ny, nz = 50, 45, 23
flights = pd.read_csv("../data_generated/fligths_spectrum_max_fuel.csv")
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
eham = nav.airport("EHAM")
ebbr = nav.airport("EBBR")
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)
ax.set_extent(plot_extent)
ax.add_feature(BORDERS, linestyle="dotted", alpha=1)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=1)


df_c = df_cost  # .assign(cost=lambda x: np.where(x.cost > 0.8, 0.8, x.cost)).assign(cost=lambda x: np.where(x.cost < 0.01, 0.01, x.cost))


norm = plt.Normalize(vmin=0.0001, vmax=0.06)
contr = ax.contourf(
    df_cost.longitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.latitude.values.reshape(nx, ny, nz)[:, :, 0],
    df_cost.cost.values.reshape(nx, ny, nz)[:, :, 6],
    levels=25,
    alpha=0.4,
    norm=norm,
    transform=trans,
    cmap="binary",
    # label = "population"
)
props = dict(boxstyle="round", facecolor="w")
ax.text(
    eham["lon"] + 0.06,
    eham["lat"] + 0.04,
    "EHAM",
    transform=trans,
    verticalalignment="top",
    bbox=props,
    alpha=0.8,
)
ax.scatter(
    eham["lon"],
    eham["lat"],
    c="k",
    s=20,
)


cmap = plt.get_cmap("Reds")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "narrow_viridis", cmap(np.linspace(0.3, 0.8, 256))
)

colors_v = cmap(np.linspace(0, 1, len(flights.num.unique())))  # Assign unique colors
norm_fid = matplotlib.colors.Normalize(vmin=0, vmax=5)  # Normalize for color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm_fid)  # For colorbar
sm.set_array([])

for i, num in enumerate(flights.num.unique()):
    flight = flights.query("num==@num")
    print(flight.mass.iloc[0] - flight.mass.iloc[-1])
    # flight0= flights0.query("num==@num")
    if i != 0:
        ax.plot(
            flight.longitude,
            flight.latitude,
            color=colors_v[(i)],
            lw=2,
            linestyle="dashed",
            transform=trans,
            label=f"Noise optimal{num}",
        )
        ax.plot(
            flight.query("cost_grid>0").longitude,
            flight.query("cost_grid>0").latitude,
            color=colors_v[(i)],
            lw=2,
            # linestyle="dashed",
            transform=trans,
            # label=f"Noise optimal{num}",
        )
    else:
        ax.plot(
            flight.longitude,
            flight.latitude,
            color=colors_v[(i)],
            lw=2,
            # linestyle="dashed",
            transform=trans,
            label=f"Noise optimal{num}",
        )

cbar = fig.colorbar(
    sm, ax=ax, orientation="horizontal", shrink=0.4, fraction=0.03, pad=0.02
)
cbar.set_ticks(np.linspace(0, 5, 5))
cbar.set_ticklabels(["0%", "0.1%", "0.5%", "1.1%", "2%"])
cbar.set_label("Fuel allowence increment")
# ax.legend()


plt.tight_layout()
plt.savefig("../figs/noise_fuel_spectrum_flights_max_fuel.png", bbox_inches="tight")
plt.show()
# # %%
# ac = "a320"
# pop_map = pd.read_parquet("../data_raw/pop_static.parquet")
# grid_type = "ll"
# treshold = 45
# table = []
# for num in flights.num.unique():
#     flight = flights.query("num==@num")
#     flight0 = flights.query("num==0")
#     pp_all, pp_uni, f = pp_affected(ac, pop_map, grid_type, flight, treshold)
#     pp_all0, pp_uni0, f0 = pp_affected(ac, pop_map, grid_type, flight0, treshold)
#     flight = flight.assign(noise=lambda x: x.cost_grid * x.thrust)
#     flight0 = flight0.assign(noise=lambda x: x.cost_grid * x.thrust)
#     table.append(
#         {
#             "id": flight.fid.unique()[0],
#             "num": flight.num.unique()[0],
#             "fuel": f.fuel.sum() / f0.fuel.sum(),
#             "cost": flight.cost_grid.sum() / flight0.cost_grid.sum(),
#             "noise": flight.noise.sum() / flight0.noise.sum(),
#             "pp_all": pp_all / pp_all0,
#             "pp_uni": pp_uni / pp_uni0,
#         }
#     )
# print(treshold)
# table = pd.DataFrame.from_dict(table)
# table

# %%
# %%
### Plotting 2 different altitude layers of df_cost
map_type = "DN"
eham = nav.airport("EHAM")
popd = (
    pd.read_parquet(f"../data_raw/D032011_1K_cropped.parquet")
    .rename(columns={"popul": "pp"})
    .fillna(0)
)
popn = (
    pd.read_parquet(f"../data_raw/N032011_1K_cropped.parquet")
    .rename(columns={"popul": "pp"})
    .fillna(0)
)
popst = pd.read_parquet("../data_raw/pop_static.parquet")
norm2 = plt.Normalize(vmin=0, vmax=14000)
proj = ccrs.TransverseMercator(
    central_longitude=eham["lon"], central_latitude=eham["lat"]
)
trans = ccrs.PlateCarree()
altp = np.linspace(0, 22000, 20)


fig = plt.figure(figsize=(6, 8))
gs = matplotlib.gridspec.GridSpec(10, 1)
ax0 = fig.add_subplot(gs[0:3, 0:1], projection=proj)
ax1 = fig.add_subplot(gs[3:6, 0:1], projection=proj)
ax2 = fig.add_subplot(gs[6:10, 0:1], projection=proj)


pop_vis = ax0.scatter(
    popd.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").lon,
    popd.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").lat,
    c=popd.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").pp,
    s=1,
    transform=trans,
    norm=norm2,
    alpha=1,
    cmap="viridis",
    label="Day map: "
    + str(
        np.round(
            popd.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").pp.sum() / 1000_000, 3
        )
    )
    + "million",
)
pop_vis = ax1.scatter(
    popn.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").lon,
    popn.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").lat,
    c=popn.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").pp,
    s=1,
    transform=trans,
    norm=norm2,
    alpha=1,
    cmap="viridis",
    label="Night map: "
    + str(
        np.round(
            popn.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").pp.sum() / 1000_000, 3
        )
    )
    + "million",
)


pop_vis = ax2.scatter(
    popst.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").lon.values,
    popst.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").lat.values,
    c=popst.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").pp.values,
    s=1,
    transform=trans,
    norm=norm2,
    alpha=1,
    cmap="viridis",
    label="Static map: "
    + str(
        np.round(
            popst.query("51.8<lat<52.5 and 4<lon<6").query("pp>0").pp.sum() / 1000_000,
            3,
        )
    )
    + "million",
)


for ax in [ax0, ax1, ax2]:
    # ax.set_extent([4, 6, 51.8, 52.5])
    ax.add_feature(BORDERS, color="k", alpha=0.3)
    ax.add_feature(COASTLINE, color="k", alpha=0.3)
    ax.scatter(eham["lon"], eham["lat"], c="r", s=10, transform=trans)
    ax.text(
        eham["lon"] + 0.1,
        eham["lat"],
        "EHAM",
        transform=trans,
        fontsize=8,
        bbox=dict(facecolor="white"),
    )
    ax.legend(fontsize=7)
    gl = ax.gridlines(draw_labels=False, color="gray", alpha=0.5, ls="--")
    gl.left_labels = True
    gl.xlocator = mticker.FixedLocator([4, 5, 6])
    gl.ylocator = mticker.FixedLocator([51.8, 52.1, 52.4])
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}
gl.bottom_labels = True

cbar2 = plt.colorbar(
    pop_vis,
    ax=ax2,
    orientation="horizontal",
    shrink=0.4,
    aspect=16,
    pad=0.12,
)
# cbar2.set_ticks([0, 0.104])
# cbar2.set_ticklabels(["Low", "High"])

cbar2.ax.set_xlabel("people per sq.km", rotation=0, labelpad=0, fontsize=7)
plt.tight_layout()
plt.savefig("../figs/compare_maps.png", bbox_inches="tight")
plt.show()

# %%
