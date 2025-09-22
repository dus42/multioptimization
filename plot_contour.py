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
from traffic.core import Traffic, Flight
from openap import Thrust, Drag
import itertools
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE, LAKES, RIVERS
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns
from traffic.visualize.markers import rotate_marker, aircraft


# %%
def interp_npd1(thrs, dis):
    npdc = pd.read_csv("npd_check.csv").query("noise_metric=='LAmax'")
    thrusts = npdc.thrust_n.to_numpy()
    dists = np.array([int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]])
    noise_levels = npdc.iloc[:, 5:15].values
    noise = np.zeros(thrs.shape)
    for i, (thr, d) in enumerate(zip(thrs, dis)):
        if d >= dists.max():
            noise[i] = 0
            continue
        elif d <= dists.min():
            d1, d2 = dists[0], dists[1]
        else:
            d1 = dists[dists < d][-1]
            d2 = dists[dists > d][0]
        if thr >= thrusts.max():
            p1, p2 = thrusts[-2], thrusts[-1]
        elif thr <= thrusts.min():
            p1, p2 = thrusts[0], thrusts[1]
        else:
            p1 = thrusts[thrusts < thr][-1]
            p2 = thrusts[thrusts > thr][0]

        L_p1_d1 = noise_levels[np.where(thrusts == p1), np.where(dists == d1)]
        L_p1_d2 = noise_levels[np.where(thrusts == p1), np.where(dists == d2)]
        L_p2_d1 = noise_levels[np.where(thrusts == p2), np.where(dists == d1)]
        L_p2_d2 = noise_levels[np.where(thrusts == p2), np.where(dists == d2)]
        L_p1_d = L_p1_d1 + (
            (L_p1_d2 - L_p1_d1)
            * (np.log10(d) - np.log10(d1))
            / (np.log10(d2) - np.log10(d1))
        )
        L_p2_d = L_p2_d1 + (
            (L_p2_d2 - L_p2_d1)
            * (np.log10(d) - np.log10(d1))
            / (np.log10(d2) - np.log10(d1))
        )
        noise[i] = L_p1_d + ((L_p2_d - L_p1_d) * (thr - p1) / (p2 - p1))
    return noise


def assign_thrust_df(f):
    drag = Drag("a320", wave_drag=True)
    thrust = Thrust("a320")
    mass = f.mass
    tas = f.tas
    alt = f.altitude
    vs = f.vertical_rate
    D = drag.clean(mass=mass, tas=tas, alt=alt, vs=vs)
    gamma = np.arctan2(vs * aero.fpm, tas * aero.kts)
    T = D + mass * 9.81 * np.sin(gamma)

    T_max = thrust.climb(tas=tas, alt=alt, roc=vs)
    T_idle = thrust.descent_idle(tas=tas, alt=alt)

    T = (
        (
            np.log(1 + np.exp(20 * (T - T_idle * 0.8) / 100_000))
            - np.log(1 + np.exp(20 * (T - T_max * 1.2) / 100_000))
        )
        / (np.log(1 + np.exp(20)))
    ) * 100_000 + T_idle * 0.8

    return f.assign(thrust=T)


# options = itertools.product(["D", "N"], [True, False])
# options = itertools.product(["D"], [False])
options = itertools.product(["D", "N"], [False])
options = pd.DataFrame(options, columns=["map_type", "is_max_fuel"])
crs_3035 = CRS.from_epsg(3035)
crs_4326 = CRS.from_epsg(4326)
transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)
eham = nav.airport("EHAM")
noise_tresholds = [55, 65, 75]
pp_dict = []
for i, (map_type, max_fuel) in options.iterrows():
    fid = "VLG12BR_219"
    df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
    df_real = (
        pd.read_parquet(f"data_generated/opensky2024_centroids_DN.parquet")
        .query("flight_id=='VLG12BR_219'")
        .reset_index(drop=True)
    )
    max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
    min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
    min_x, min_y = transformer_xy.transform(*min_lola)
    max_x, max_y = transformer_xy.transform(*max_lola)

    pop_map0 = pd.read_parquet(f"data_raw/{map_type}032011_1K_cropped.parquet").rename(
        columns={"popul": "pp"}
    )

    pop_map = pop_map0.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
    pop_x = pop_map.x.unique()
    pop_y = pop_map.y.unique()
    X2d, Y2d, Z2d = np.meshgrid(pop_x, pop_y, [0])
    point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T
    npdc = pd.read_csv("npd_check.csv").query("noise_metric=='LAmax'")
    thrusts = npdc.thrust_n.to_numpy()
    dists = [int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]]
    noise_levels = npdc.iloc[:, 5:15].values
    interp_npd = RegularGridInterpolator(
        (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
    )

    ################################################################
    #######################################
    flight = pd.read_csv(f"data_generated/200_optimal_flights_new.csv").query(
        "max_fuel==@max_fuel and map_type==@map_type and obj=='pop'"
    )
    flight = assign_thrust_df(flight)

    [flonmin, flonmax, flatmin, flatmax] = [4.2, 5.3, 51.6, 52.5]
    pp_total = pop_map.query(
        f"{flonmin}<lon<{flonmax} and {flatmin} < lat<{flatmax}"
    ).pp.sum()
    xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
    point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 25000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    num = len(flight)
    for i in range(num):
        thr = flight.thrust.values[i]
        # ns = interp_npd(np.array([np.array([thr] * len(dist[i])), dist[i]]).T)
        ns = interp_npd1(np.array([thr] * len(dist[i])), dist[i])

        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    # aggregated_noise = np.sum(noise.reshape(num, len(pop_y), -1), axis=0)
    noise1 = noise.reshape(num, len(pop_y), -1)
    population = pop_map.pp.values.reshape(len(pop_y), len(pop_x))

    for treshold in noise_tresholds:
        affected_pop = np.zeros(dist_0.shape)
        for i in range(num):
            affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

        # agg_pop = np.sum(affected_pop.reshape(num, len(pop_y), -1), axis=0)
        # people_affected = sum(sum(agg_pop))
        people_affected = (
            np.where(
                np.sum(affected_pop.reshape(num, len(pop_y), -1), axis=0) > 0,
                population.reshape(len(pop_y), -1),
                0,
            )
        ).sum()
        pp_dict.append(
            {
                "fid": fid,
                "people_affected": people_affected,
                "pp_total": pp_total,
                "map_type": map_type,
                "max_fuel": max_fuel,
                "treshold": treshold,
            }
        )
    proj = ccrs.PlateCarree()
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
    ax.set_extent([4, 5.5, 51.4, 52.65])

    ax.add_feature(BORDERS, linestyle="dotted", alpha=0.6)
    ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.6)
    gl = ax.gridlines(draw_labels=False, color="gray", alpha=0.5, ls="--")
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {
        "size": 6,
    }
    gl.ylabel_style = {
        "size": 6,
    }

    ax.plot(
        [flonmax, flonmax, flonmin, flonmin, flonmax],
        [flatmin, flatmax, flatmax, flatmin, flatmin],
        transform=trans,
        c="k",
        linestyle="dashed",
        alpha=0.5,
    )
    airports["EHAM"].runways["18L"].plot(ax=ax, alpha=0.5)

    pop_filtered = pop_map0.query("10 < pp")

    # Normalize pp to [0, 1] with 10 mapped to 0, 2000+ mapped to 1
    pp_alpha = (pop_filtered["pp"] - 10) / (20000 - 10)
    pp_alpha = np.clip(pp_alpha, 0, 1)  # clamp between 0 and 1

    # Set constant color (black) and variable alpha
    colors = [mcolors.to_rgba("tab:blue", alpha=a) for a in pp_alpha]
    ax.scatter(
        pop_filtered.lon,
        pop_filtered.lat,
        c=colors,
        s=5,
        marker="s",
        transform=trans,
        edgecolors="none",
    )

    cmap = plt.get_cmap("CMRmap_r")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "narrow_gnuplot", cmap(np.linspace(0.5, 1, 256))
    )

    pop_lon, pop_lat = transformer_ll.transform(X2d, Y2d)

    noise_n = np.max(noise1, axis=0)
    norm = plt.Normalize(vmin=55, vmax=70, clip=True)
    cs = ax.contour(
        pop_lon[:, :, 0],
        pop_lat[:, :, 0],
        noise_n,
        norm=norm,
        cmap=cmap,
        levels=[55, 65, 75],
        transform=trans,
        linewidth=2,
        alpha=1,
    )
    labels = ax.clabel(
        cs,
        fontsize=10,
    )
    for label in labels:
        label.set_fontweight("bold")
        label.set_fontfamily("monospace")

    lonf, latf = transformer_ll.transform(xf, yf)

    ax.plot(
        lonf,
        latf,
        lw=2,
        c="k",
        transform=trans,
        alpha=0.5,
    )
    # ax.scatter(lonf,latf,s=1, c="k", transform=trans)
    norm = plt.Normalize(vmin=-10, vmax=100, clip=True)
    if max_fuel:
        plt.savefig(
            f"figs/contours_n_{map_type}_max_fuel.png",
            pad_inches=0,
            bbox_inches="tight",
            dpi=500,
        )
    else:
        plt.savefig(
            f"figs/contours_n_{map_type}.png",
            pad_inches=0,
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()

df_pp = pd.DataFrame.from_dict(pp_dict)
# pp_total = pop_map.query(
#     f"{flonmin}<lon<{flonmax} and {flatmin} < lat<{flatmax}"
# ).pp.sum()
# df_pp = df_pp.assign(pp_total=pp_total)
df_pp = df_pp.assign(percentage=lambda x: x.people_affected / x.pp_total * 100)
df_pp


# %%%
def nd(val):
    if val == "N":
        return "Night"
    else:
        return "Day"


df = df_pp.sort_values(by=["map_type", "percentage"], ascending=False)
df["treshold"] = df["treshold"].apply(lambda x: str(x) + " dBA")
# df["percentage"] = df["percentage"].apply(lambda x: str(round(x,2))+" %")
df["map_type"] = df["map_type"].apply(nd)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(df, x="treshold", y="percentage", hue="map_type", ax=ax, errorbar=None)
ax.set_xlabel("")
ax.set_ylabel("People affected %")
ax.legend(title=None)


# %%


def perc(val):
    return f"{val:.3f}~\%"


def db(val):
    return f"{val}~dBA"


df_small = df_pp.pivot_table(
    index=["treshold", "max_fuel"], columns="map_type", values="percentage"
).reset_index()
df_small["D"] = df_small["D"].apply(perc)
df_small["N"] = df_small["N"].apply(perc)
df_small["treshold"] = df_small["treshold"].apply(db)
df_small.columns.name = None
df_small.rename(columns={"D": "Day", "N": "Night"}, inplace=True)

df_small[["Day", "Night"]] = df_small[["Day", "Night"]].round(2)


print(
    df_small.sort_values(by=["max_fuel"])[
        ["max_fuel", "treshold", "Day", "Night"]
    ].to_latex(index=False)
)

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# cmap = plt.cm.viridis.copy()
# flight = pd.read_csv(f"data_generated/flights_noise_opt_{map_type}.csv").query(
#     "fid==fid.iloc[0]"
# )
# Z = noise1.reshape(151, len(pop_y), -1)

Z = np.max(noise1, axis=0)
Z2 = affected_pop.reshape(151, len(pop_y), -1)[:145]
# Create a 3D plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
zf = flight.iloc[:145, :].altitude.values * aero.ft
fsc = flight.iloc[:145, :].query("cost_grid==0")
zsc = fsc.altitude.values * aero.ft
xfsc, yfsc = transformer_xy.transform(fsc.longitude, fsc.latitude)
xf, yf = transformer_xy.transform(
    flight.iloc[:145, :].longitude, flight.iloc[:145, :].latitude
)
z_values = range(0, 110, 8)

norm = plt.Normalize(vmin=45, vmax=75, clip=True)
cntr = ax.contourf(
    pop_x,
    pop_y,
    Z[:, :],
    levels=np.arange(35, 95, 10),
    # levels=[55,65,75],
    # 200,
    zdir="z",
    offset=0,  # zf[z],
    cmap="viridis",
    alpha=1,
    norm=norm,
    zorder=0,
)

sctr = ax.scatter3D(
    pop_filtered.query(
        "pp>2000 and @pop_x.min()<x<@pop_x.max() and @pop_y.min()<y<@pop_y.max()"
    ).x,
    pop_filtered.query(
        "pp>2000 and @pop_x.min()<x<@pop_x.max() and @pop_y.min()<y<@pop_y.max()"
    ).y,
    0,
    s=0.1,
    c="k",
    alpha=0.4,
    norm=norm,
    zorder=100,
)
# ax.scatter(
#     pop_filtered.lon,
#     pop_filtered.lat,
#     c=colors,
#     s=5,
#     marker="s",
#     transform=trans,
#     edgecolors="none",
# )
ax.set_proj_type("persp", focal_length=0.3)
ax.view_init(elev=30, azim=-25, roll=0)
Z2 = Z
Z2 = np.zeros(Z.shape)
# for z in z_values:
#     ax.contourf(
#         pop_x,
#         pop_y,
#         Z2[:, :],
#         0,
#         zdir="z",
#         offset=zf[z],
#         cmap="viridis",
#         alpha=0.06,
#         norm=plt.Normalize(0, 1),
#         zorder=10
#     )


cbar = plt.colorbar(
    cntr, shrink=0.4, orientation="horizontal", pad=-0.001, anchor=(0.7, 1)
)
# cbar.set_ticks(np.arange(0.05, 102.05, 20))
cbar.set_ticks(np.arange(45, 85.0, 10))
l = []
for i in np.arange(35, 80, 10)[1:]:
    l.append(str(i))
# cbar.set_ticklabels(["Low", "High"])
cbar.set_ticklabels(l)

cbar.ax.set_xlabel(r"$L_{A_{max}}$ dBA", rotation=0, labelpad=5)

yticks0 = ax.get_yticks()
xticks0 = ax.get_xticks()
ax.set_xticks(pop_x)
ax.set_yticks(pop_y)
yticks = ax.get_yticks()
xticks = ax.get_xticks()

# ll_min = transformer_ll.transform(xticks,yticks)
# xy_min = transformer_xy.transform(np.linspace(ll_min[0].min(),ll_min[0].max(),3),np.round(np.linspace(ll_min[1].min(),ll_min[1].max(),3),2))
# ax.set_xticks(xy_min[0])
# ax.set_yticks(xy_min[1])
# ax.set_xticklabels(np.round(np.linspace(ll_min[0].min(),ll_min[0].max(),3),2))
# ax.set_yticklabels(np.round(np.linspace(ll_min[1].min(),ll_min[1].max(),3),2))
# ax.set_xticklabels(np.linspace(ll_min[0].min(),ll_min[0].max(),3))
# ax.set_yticklabels(np.linspace(ll_min[1].min(),ll_min[1].max(),3))

zticks = ax.get_zticks()
ax.set_xlim(pop_x.min(), pop_x.max())
ax.set_ylim(pop_y.min(), pop_y.max())
ax.set_zlim(zticks.min(), zticks.max())
yticks = ax.get_yticks()
xticks = ax.get_xticks()


# ll_min = transformer_ll.transform(xticks,yticks)
# xy_min = transformer_xy.transform(np.linspace(ll_min[0].min(),ll_min[0].max(),3),np.round(np.linspace(ll_min[1].min(),ll_min[1].max(),3),2))
# ax.set_xticks(xy_min[0])
# ax.set_yticks(xy_min[1])
# ax.set_xticklabels(np.round(np.linspace(ll_min[0].min(),ll_min[0].max(),3),2))
# ax.set_yticklabels(np.round(np.linspace(ll_min[1].min(),ll_min[1].max(),3),2))
# ax.set_xticklabels(np.linspace(ll_min[0].min(),ll_min[0].max(),3))
# ax.set_yticklabels(np.linspace(ll_min[1].min(),ll_min[1].max(),3))
ax.set_xticks([])
ax.set_yticks([])

ax.scatter3D(xf[:65], yf[:65], zf[:65], s=1, c="tab:red", zorder=30)
ax.scatter3D(xfsc, yfsc, zsc, s=1, c="tab:blue", zorder=30)
ax.scatter(
    xfsc[-1],
    yfsc[-1],
    zsc[-1],
    marker=rotate_marker(aircraft, 270),
    s=100,
    c="tab:blue",
)
ax.plot3D(xf, yf, 0, c="k", linestyle="--", linewidth=1, alpha=0.5, zorder=30)
ax.plot3D(
    [pop_x.min(), pop_x.max()],
    [pop_y.max(), pop_y.max()],
    [zsc[0], zsc[0]],
    linewidth=1,
    linestyle="--",
    alpha=0.2,
    c="k",
)
ax.plot3D(
    [pop_x.min(), pop_x.min()],
    [pop_y.min(), pop_y.max()],
    [zsc[0], zsc[0]],
    linewidth=1,
    linestyle="--",
    alpha=0.2,
    c="k",
)

for xx in np.arange(0, len(xf), 5):
    ax.plot3D(
        [xf[xx], xf[xx]],
        [yf[xx], yf[xx]],
        [0, zf[xx]],
        c="k",
        linestyle="--",
        linewidth=1,
        alpha=0.2,
        zorder=20,
    )


ax.plot3D(
    [xfsc[0], xfsc[0]],
    [yfsc[0], yfsc[0]],
    [0, zsc[0]],
    c="navy",
    linestyle="--",
    alpha=0.4,
)
ax.plot3D(
    [xfsc[0], xfsc[0]],
    [yticks.max(), yfsc[0]],
    [zsc[0], zsc[0]],
    c="navy",
    linestyle="--",
    alpha=0.4,
)
ax.plot3D(
    [xticks.min(), xfsc[0]],
    [yfsc[0], yfsc[0]],
    [zsc[0], zsc[0]],
    c="navy",
    linestyle="--",
    alpha=0.4,
)


ax.scatter3D(xfsc[0], yfsc[0], zsc[0], c="navy", s=20, marker="*", zorder=60, alpha=0.7)
ax.scatter3D(xfsc[0], yfsc[0], 0, c="navy", s=40, marker="*", zorder=20, alpha=1)
ax.scatter3D(
    xfsc[0], yticks.max(), zsc[0], c="navy", s=40, marker="*", zorder=20, alpha=0.4
)
ax.scatter3D(
    xticks.min(), yfsc[0], zsc[0], c="navy", s=40, marker="*", zorder=20, alpha=0.4
)

# ax.scatter3D(xfsc[0], yfsc[0], 0, c="k", s=100)

# ax.contourf(
#     pop_x,
#     pop_y,
#     Z2[:, :],
#     levels=2,
#     zdir="z",
#     offset=zsc[0],
#     cmap="cividis",
#     alpha=0.2,
#     zorder=10
#     # norm=plt.Normalize(vmin=0, vmax=90, clip=True),
# )

ax.set_zticks(np.arange(0, 35000 * aero.ft, 5000 * aero.ft))
zticks = ax.get_zticks()
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
# ax.set_ylabel("Lat")
ax.set_xlabel("Longitude", labelpad=-10, fontsize=10)
ax.set_ylabel("Latitude", labelpad=-15, fontsize=10)
ax.set_zlabel("Altitude, kft")
plt.tight_layout()
# ax.set_title("3D Contour Plot with Layers")
plt.savefig(f"figs/3dplot.png", pad_inches=0.3, bbox_inches="tight", dpi=300)
plt.show()
# %% Plot real contour
from traffic.core import Flight, Traffic
from glob import glob
from openap import aero, Thrust, prop, FuelFlow, Drag


def find_arrival(flight):
    flights_info = pd.read_csv(
        "data_raw/flights_info.csv"
    )  # [["icao24","departure","arrival","callsign","day"]]
    flights_info = flights_info.query("arrival in @airports.data.icao")
    flights_info["day"] = pd.to_datetime(flights_info["day"]).dt.tz_localize(None)
    flights_info["firstseen"] = pd.to_datetime(
        flights_info["firstseen"]
    ).dt.tz_localize(None)
    df = flight.data
    month = df.timestamp.min().month
    day = df.timestamp.min().day
    finfo = flights_info.query(f"day.dt.month=={month} and day.dt.day=={day}")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    min_time = df["timestamp"].min()
    finfo = finfo.assign(dif_time=lambda x: (x["firstseen"] - min_time).abs()).query(
        "dif_time == dif_time.min()"
    )
    df = df.merge(
        finfo[["icao24", "departure", "arrival", "callsign"]],
        on=["icao24", "callsign"],
        how="left",
    )
    return Flight(df)


mass_estim_coefs = np.array([1.32, -2.21, 155, 73360.0])


def drop_dups(f):
    return f.drop_duplicates(subset=["timestamp"], keep="first").drop_duplicates(
        subset=["latitude", "longitude"], keep="first"
    )


def assign_tow(f):
    m_mtow = prop.aircraft("a320")["limits"]["MTOW"]
    oew = prop.aircraft("a320")["limits"]["OEW"]
    mass_estim_coefs = np.array([1.32, -2.21, 155, 73360])
    arrival = f.data.arrival.values[0]
    if type(arrival) is type("string") and arrival != "NaN":
        if arrival == "LDZK":
            distance = (
                aero.distance(
                    airports["EHAM"].latitude,
                    airports["EHAM"].longitude,
                    46.012778,
                    15.860000,
                )
                / 1000
            )
        else:
            distance = (
                aero.distance(
                    airports["EHAM"].latitude,
                    airports["EHAM"].longitude,
                    airports[arrival].latitude,
                    airports[arrival].longitude,
                )
                / 1000
            )
        # print("distance to "+arrival+" - "+str(distance)+" km" )
    else:
        distance = 1500
        # print(f.flight_id + "nan arrival")
    tow = (
        mass_estim_coefs[0] * distance
        + mass_estim_coefs[1] * f.data.mean_alt.values[0]
        + mass_estim_coefs[2] * f.data.mean_speed.values[0]
        + mass_estim_coefs[3]
    )
    if tow > m_mtow:
        tow = m_mtow
    elif tow < oew:
        tow = oew * 1.2
    return f.assign(tow=tow)


def assign_fuel(flight):
    df = (
        flight.fuelflow(initial_mass=flight.data.tow.values[0])
        .data.assign(fuel=lambda x: x.fuelflow * x.dt)
        .assign(fuel_sum=lambda x: x.fuel.sum())
    )
    return Flight(df)


def assign_mass(f):
    return f.assign(mass=lambda x: x.tow - x.fuel.cumsum())
    # fuelflow.enroute(f.tow, f.groundspeed, f.fl*100, f.vertical_rate)


def assign_thrust(f):
    drag = Drag("a320", wave_drag=True)
    thrust = Thrust("a320")
    mass = f.data.mass
    tas = f.data.groundspeed
    alt = f.data.altitude
    vs = f.data.vertical_rate
    D = drag.clean(mass=mass, tas=tas, alt=alt, vs=vs)
    gamma = np.arctan2(vs * aero.fpm, tas * aero.kts)
    T = D + mass * 9.81 * np.sin(gamma)

    T_max = thrust.climb(tas=tas, alt=alt, roc=vs)
    T_idle = thrust.descent_idle(tas=tas, alt=alt)

    T = (
        (
            np.log(1 + np.exp(20 * (T - T_idle * 0.8) / 100_000))
            - np.log(1 + np.exp(20 * (T - T_max * 1.2) / 100_000))
        )
        / (np.log(1 + np.exp(20)))
    ) * 100_000 + T_idle * 0.8

    return f.assign(thrust=T)


map_type = "D"
t = Traffic(
    pd.read_parquet(f"data_raw/opensky/monthly/2024-12.parquet")
    .query("flight_id=='VLG12BR_219'")
    .reset_index(drop=True)
)
t = (
    t.pipe(drop_dups)
    .pipe(find_arrival)
    .pipe(assign_tow)
    .pipe(assign_fuel)  # not needed
    .pipe(assign_mass)
    .pipe(assign_thrust)
    .resample(151)  # not needed
    .eval(1, desc="preprocessing")
)
t = t.query("distance<220").compute_xy()
t = t.assign(
    x=lambda df: df.x.astype(int),
    y=lambda df: df.y.astype(int),
)

df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
df_real = (
    pd.read_parquet(f"data_generated/opensky2024_centroids_DN.parquet")
    .query("flight_id=='VLG12BR_219'")
    .reset_index(drop=True)
)
max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
min_x, min_y = transformer_xy.transform(*min_lola)
max_x, max_y = transformer_xy.transform(*max_lola)

pop_map0 = pd.read_parquet(f"data_raw/{map_type}032011_1K_cropped.parquet").rename(
    columns={"popul": "pp"}
)

pop_map = pop_map0.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
pop_x = pop_map.x.unique()
pop_y = pop_map.y.unique()
X2d, Y2d, Z2d = np.meshgrid(pop_x, pop_y, [0])
point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T


npdc = pd.read_csv("npd_check.csv").query("noise_metric=='LAmax'")
thrusts = npdc.thrust_n.to_numpy()
dists = [int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]]
noise_levels = npdc.iloc[:, 5:15].values
interp_npd = RegularGridInterpolator(
    (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
)

################################################################
#######################################

flight_r = t[0].data.assign(h=lambda x: x.altitude * aero.ft)

[flonmin, flonmax, flatmin, flatmax] = [4.2, 5.3, 51.6, 52.5]
pp_total = pop_map.query(
    f"{flonmin}<lon<{flonmax} and {flatmin} < lat<{flatmax}"
).pp.sum()
xf, yf = transformer_xy.transform(flight_r.longitude, flight_r.latitude)
point3d = np.array([xf, yf, flight_r.h.values]).reshape(3, -1).T
dist = distance_matrix(point3d, point2d)
dist_0 = np.where(dist > 25000 * aero.ft, -100, dist)

noise = np.zeros(dist_0.shape)
num = len(flight_r)
for i in range(num):
    thr = flight_r.thrust.values[i]
    # ns = interp_npd(np.array([np.array([thr] * len(dist[i])), dist[i]]).T)
    ns = interp_npd1(np.array([thr] * len(dist[i])), dist[i])
    noise[i] = np.where(dist_0[i] < 0, 0, ns)
noise1 = noise.reshape(num, len(pop_y), -1)
population = pop_map.pp.values.reshape(len(pop_y), len(pop_x))

for treshold in noise_tresholds:
    affected_pop = np.zeros(dist_0.shape)
    for i in range(num):
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    people_affected = (
        np.where(
            np.sum(affected_pop.reshape(num, len(pop_y), -1), axis=0) > 0,
            population.reshape(len(pop_y), -1),
            0,
        )
    ).sum()
    pp_dict.append(
        {
            "fid": fid,
            "people_affected": people_affected,
            "pp_total": pp_total,
            "map_type": "real",
            "max_fuel": "real",
            "treshold": treshold,
        }
    )
proj = ccrs.PlateCarree()
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
ax.set_extent([4, 5.5, 51.4, 52.65])

ax.add_feature(BORDERS, linestyle="dotted", alpha=0.6)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.6)
# ax.add_feature(LAKES, linestyle="dotted", alpha=0.6)
# ax.add_feature(RIVERS, linestyle="dotted", alpha=0.6)
gl = ax.gridlines(draw_labels=False, color="gray", alpha=0.5, ls="--")
gl.bottom_labels = True
gl.left_labels = True
gl.xlabel_style = {
    "size": 6,
}
gl.ylabel_style = {
    "size": 6,
}

ax.plot(
    [flonmax, flonmax, flonmin, flonmin, flonmax],
    [flatmin, flatmax, flatmax, flatmin, flatmin],
    transform=trans,
    c="k",
    linestyle="dashed",
    alpha=0.5,
)

pop_filtered = pop_map0.query("10 < pp")

pp_alpha = (pop_filtered["pp"] - 10) / (20000 - 10)
pp_alpha = np.clip(pp_alpha, 0, 1)  # clamp between 0 and 1

colors = [mcolors.to_rgba("tab:blue", alpha=a) for a in pp_alpha]
ax.scatter(
    pop_filtered.lon,
    pop_filtered.lat,
    c=colors,
    s=5,
    marker="s",
    transform=trans,
    edgecolors="none",
)

cmap = plt.get_cmap("CMRmap_r")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "narrow_gnuplot", cmap(np.linspace(0.5, 1, 256))
)

pop_lon, pop_lat = transformer_ll.transform(X2d, Y2d)

noise_n = np.max(noise1, axis=0)
norm = plt.Normalize(vmin=55, vmax=70, clip=True)
cs = ax.contour(
    pop_lon[:, :, 0],
    pop_lat[:, :, 0],
    noise_n,
    norm=norm,
    cmap=cmap,
    levels=[55, 65, 75],
    transform=trans,
    linewidth=2,
    alpha=1,
)
labels = ax.clabel(
    cs,
    fontsize=10,
)
for label in labels:
    label.set_fontweight("bold")
    label.set_fontfamily("monospace")


lonf, latf = transformer_ll.transform(xf, yf)

ax.plot(
    lonf,
    latf,
    lw=2,
    c="k",
    transform=trans,
    alpha=0.5,
)
# ax.scatter(lonf, latf, s=1, c="k", transform=trans)
norm = plt.Normalize(vmin=-10, vmax=100, clip=True)
if max_fuel:
    plt.savefig(
        f"figs/contours_n_{map_type}_real.png",
        pad_inches=0,
        bbox_inches="tight",
        dpi=500,
    )
else:
    plt.savefig(
        f"figs/contours_n_{map_type}_real.png",
        pad_inches=0,
        bbox_inches="tight",
        dpi=500,
    )
plt.show()

df_pp = pd.DataFrame.from_dict(pp_dict)

df_pp = df_pp.assign(percentage=lambda x: x.people_affected / x.pp_total * 100)
df_pp


# %%
def nd(val):
    if val == "N":
        return "Night"
    if val == "real":
        return "Real"
    else:
        return "Day"


def order(val):
    if val == "N":
        return 1
    if val == "real":
        return 3
    else:
        return 2


df = df_pp.query("max_fuel!=True")
df["order"] = df["map_type"].apply(order)
df["treshold"] = df["treshold"].apply(lambda x: str(x) + " dBA")
# df["percentage"] = df["percentage"].apply(lambda x: str(round(x,2))+" %")
df["map_type"] = df["map_type"].apply(nd)
df = df.sort_values(by="order")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(df, x="treshold", y="percentage", hue="map_type", ax=ax, errorbar=None)
ax.set_xlabel("")
ax.set_ylabel("People affected %")
ax.legend(title=None)
# ax.grid(True)
plt.savefig(
    f"figs/barchart_DN.png",
    pad_inches=0,
    bbox_inches="tight",
    dpi=500,
)
# # %%Runway


# t = Traffic(
#     pd.read_parquet(f"data_generated/opensky2024_runway18L_DN.parquet").reset_index(
#         drop=True
#     )
# )
# # t = (
# #     t.pipe(drop_dups)
# #     .pipe(find_arrival)
# #     .pipe(assign_tow)
# #     .pipe(assign_fuel)#not needed
# #     .pipe(assign_mass)
# #     .pipe(assign_thrust)#not needed
# #     .eval(1, desc="preprocessing")
# # )
# # t = t.query("distance<220").compute_xy()
# # t = t.assign(
# #     x=lambda df: df.x.astype(int),
# #     y=lambda df: df.y.astype(int),
# # )
# map_type="D"
# df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
# df_real = (
#     pd.read_parquet(f"data_generated/opensky2024_centroids_DN.parquet")
#     .query("flight_id=='VLG12BR_219'")
#     .reset_index(drop=True)
# )
# max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
# min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
# min_x, min_y = transformer_xy.transform(*min_lola)
# max_x, max_y = transformer_xy.transform(*max_lola)

# pop_map0 = pd.read_parquet(f"data_raw/{map_type}032011_1K_cropped.parquet").rename(
#     columns={"popul": "pp"}
# )

# pop_map = pop_map0.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
# pop_x = pop_map.x.unique()
# pop_y = pop_map.y.unique()
# X2d, Y2d, Z2d = np.meshgrid(pop_x, pop_y, [0])
# point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T


# npdc = pd.read_csv("npd_check.csv").query("noise_metric=='LAmax'")
# thrusts = npdc.thrust_n.to_numpy()
# dists = [int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]]
# noise_levels = npdc.iloc[:, 5:15].values
# interp_npd = RegularGridInterpolator(
#     (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
# )

# ################################################################
# #######################################

# flight_r = t.data.assign(h=lambda x: x.altitude * aero.ft)

# [flonmin, flonmax, flatmin, flatmax] = [4.2, 5.3, 51.6, 52.5]
# pp_total = pop_map.query(
#     f"{flonmin}<lon<{flonmax} and {flatmin} < lat<{flatmax}"
# ).pp.sum()
# xf, yf = transformer_xy.transform(flight_r.longitude, flight_r.latitude)
# point3d = np.array([xf, yf, flight_r.h.values]).reshape(3, -1).T
# dist = distance_matrix(point3d, point2d)
# dist_0 = np.where(dist > 25000 * aero.ft, -100, dist)

# noise = np.zeros(dist_0.shape)
# num = len(flight_r)
# for i in range(num):
#     thr = flight_r.thrust.values[i]
#     ns = interp_npd(np.array([np.array([thr] * len(dist[i])), dist[i]]).T)

#     noise[i] = np.where(dist_0[i] < 0, 0, ns)
# noise1 = noise.reshape(num, len(pop_y), -1)
# population = pop_map.pp.values.reshape(len(pop_y), len(pop_x))

# for treshold in noise_tresholds:
#     affected_pop = np.zeros(dist_0.shape)
#     for i in range(num):
#         affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

#     people_affected = (
#         np.where(
#             np.sum(affected_pop.reshape(num, len(pop_y), -1), axis=0) > 0,
#             population.reshape(len(pop_y), -1),
#             0,
#         )
#     ).sum()
#     pp_dict.append(
#         {
#             "fid": fid,
#             "people_affected": people_affected,
#             "pp_total": pp_total,
#             "map_type": "real",
#             "max_fuel": "real",
#             "treshold": treshold,
#         }
#     )
# proj = ccrs.PlateCarree()
# proj = ccrs.TransverseMercator(
#     central_longitude=eham["lon"], central_latitude=eham["lat"]
# )
# trans = ccrs.PlateCarree()
# fig, ax = plt.subplots(
#     1,
#     1,
#     figsize=(6, 6),
#     subplot_kw=dict(projection=proj),
# )
# ax.set_extent([4, 5.5, 51.4, 52.65])

# ax.add_feature(BORDERS, linestyle="dotted", alpha=0.6)
# ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.6)
# ax.add_feature(LAKES, linestyle="dotted", alpha=0.6)
# ax.add_feature(RIVERS, linestyle="dotted", alpha=0.6)
# gl = ax.gridlines(draw_labels=False, color="gray", alpha=0.5, ls="--")
# gl.bottom_labels = True
# gl.left_labels = True
# gl.xlabel_style = {
#     "size": 6,
# }
# gl.ylabel_style = {
#     "size": 6,
# }

# ax.plot(
#     [flonmax, flonmax, flonmin, flonmin, flonmax],
#     [flatmin, flatmax, flatmax, flatmin, flatmin],
#     transform=trans,
#     c="k",
#     linestyle="dashed",
#     alpha=0.5,
# )

# pop_filtered = pop_map0.query("10 < pp")

# pp_alpha = (pop_filtered["pp"] - 10) / (20000 - 10)
# pp_alpha = np.clip(pp_alpha, 0, 1)  # clamp between 0 and 1

# colors = [mcolors.to_rgba("tab:blue", alpha=a) for a in pp_alpha]
# ax.scatter(
#     pop_filtered.lon,
#     pop_filtered.lat,
#     c=colors,
#     s=5,
#     marker="s",
#     transform=trans,
#     edgecolors="none",
# )

# norm = plt.Normalize(vmin=0, vmax=1, clip=True)
# pop_lon, pop_lat = transformer_ll.transform(X2d, Y2d)
# # noise45 = np.sum(noise1 > 45, axis=0)

# # cmap = ["hsv_r", "Purples_r", "Wistia_r", "Greens_r"]
# # colors = ["r", "purple", "darkorange", "darkgreen"]
# cmap = ["hsv_r", "Purples_r", "Greens_r"]
# colors = ["r", "purple", "darkgreen"]
# manual_positions = [
#     # 45: [(4.75, 51.2)],
#     [transformer_xy.transform(4.8, 51.8)],
#     [transformer_xy.transform(4.81, 52.0)],
#     [transformer_xy.transform(4.8, 52.2)],
# ]
# for i, db in enumerate(noise_tresholds):
#     # noise_n = gaussian_filter(noise_n, sigma=0.1)
#     noise_n = np.sum(noise1 > db, axis=0)
#     cs = ax.contour(
#         pop_lon[:, :, 0],
#         pop_lat[:, :, 0],
#         noise_n,
#         norm=norm,
#         cmap=cmap[i],
#         levels=0,
#         transform=trans,
#         linewidth=2,
#         alpha=1,
#         # label = db,
#     )
#     labels = ax.clabel(
#         cs,
#         fmt={0: f"{db}"},  # Force label to show the dB threshold
#         fontsize=10,
#         inline=True,
#         colors=colors[i],
#         # manual=manual_positions[i],
#     )
#     for label in labels:
#         label.set_fontweight("bold")
#         label.set_fontfamily("monospace")
# lonf, latf = transformer_ll.transform(xf, yf)

# # ax.plot(
# #     lonf,
# #     latf,
# #     lw=2,
# #     c="k",
# #     transform=trans,
# #     alpha=0.5,
# # )
# ax.scatter(lonf,latf,s=1, c="k", transform=trans)
# norm = plt.Normalize(vmin=-10, vmax=100, clip=True)
# if max_fuel:
#     plt.savefig(
#         f"figs/contours_n_{map_type}_real.png",
#         pad_inches=0,
#         bbox_inches="tight",
#         dpi=500,
#     )
# else:
#     plt.savefig(
#         f"figs/contours_n_{map_type}_real.png",
#         pad_inches=0,
#         bbox_inches="tight",
#         dpi=500,
#     )
# plt.show()

# df_pp = pd.DataFrame.from_dict(pp_dict)


# df_pp = df_pp.assign(percentage=lambda x: x.people_affected / x.pp_total * 100)
# df_pp
# %%
def nd(val):
    if val == "N":
        return "Night"
    if val == "real":
        return "Real"
    else:
        return "Day"


def order(val):
    if val == "N":
        return 3
    if val == "real":
        return 1
    else:
        return 2


palette = {"Night": "tab:blue", "Day": "tab:orange", "Real": "tab:green"}


df = df_pp.query("max_fuel==False or max_fuel == 'real'")
df["order"] = df["map_type"].apply(order)
df["treshold"] = df["treshold"].apply(lambda x: str(x) + " dBA")
# df["percentage"] = df["percentage"].apply(lambda x: str(round(x,2))+" %")
df["map_type"] = df["map_type"].apply(nd)
df = df.sort_values(by="order")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(
    df,
    x="treshold",
    y="percentage",
    hue="map_type",
    ax=ax,
    errorbar=None,
    palette=palette,
)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("")
ax.set_ylabel("People affected %")
ax.legend(title=None)


# %%
def perc(val):
    return f"{val:.3f}~\%"


def db(val):
    return f"{val}~dBA"


df_small = df_pp.pivot_table(
    index=["treshold"], columns="map_type", values="percentage"
).reset_index()
df_small["D"] = df_small["D"].apply(perc)
df_small["N"] = df_small["N"].apply(perc)
df_small["real"] = df_small["real"].apply(perc)
df_small["treshold"] = df_small["treshold"].apply(db)
df_small.columns.name = None
df_small.rename(columns={"D": "Day", "N": "Night", "real": "Real"}, inplace=True)

df_small[["Day", "Night"]] = df_small[["Day", "Night"]].round(2)


print(
    df_small[
        [
            "treshold",
            "Real",
            "Day",
            "Night",
        ]
    ].to_latex(index=False)
)

# %%
# # %%
# # t=Traffic(pd.read_parquet(f"data_generated/opensky2024_runway18L_DN.parquet").reset_index(drop=True))
# t = pd.read_csv("data_generated/optimal_flights_runway18L_DN.csv").query(
#     "max_fuel == True"
# )
# # t = (
# #     t.pipe(drop_dups)
# #     .pipe(find_arrival)
# #     .pipe(assign_tow)
# #     .pipe(assign_fuel)#not needed
# #     .pipe(assign_mass)
# #     .pipe(assign_thrust)#not needed
# #     .eval(1, desc="preprocessing")
# # )
# # t = t.query("distance<220").compute_xy()
# # t = t.assign(
# #     x=lambda df: df.x.astype(int),
# #     y=lambda df: df.y.astype(int),
# # )

# df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
# df_real = (
#     pd.read_parquet(f"data_generated/opensky2024_centroids_DN.parquet")
#     .query("flight_id=='VLG12BR_219'")
#     .reset_index(drop=True)
# )
# max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
# min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
# min_x, min_y = transformer_xy.transform(*min_lola)
# max_x, max_y = transformer_xy.transform(*max_lola)

# pop_map0 = pd.read_parquet(f"data_raw/{map_type}032011_1K_cropped.parquet").rename(
#     columns={"popul": "pp"}
# )

# pop_map = pop_map0.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
# pop_x = pop_map.x.unique()
# pop_y = pop_map.y.unique()
# X2d, Y2d, Z2d = np.meshgrid(pop_x, pop_y, [0])
# point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T


# npdc = pd.read_csv("npd_check.csv").query("noise_metric=='LAmax'")
# thrusts = npdc.thrust_n.to_numpy()
# dists = [int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]]
# noise_levels = npdc.iloc[:, 5:15].values
# interp_npd = RegularGridInterpolator(
#     (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
# )

# ################################################################
# #######################################

# flight_r = t

# [flonmin, flonmax, flatmin, flatmax] = [4.2, 5.3, 51.6, 52.5]
# pp_total = pop_map.query(
#     f"{flonmin}<lon<{flonmax} and {flatmin} < lat<{flatmax}"
# ).pp.sum()
# xf, yf = transformer_xy.transform(flight_r.longitude, flight_r.latitude)
# point3d = np.array([xf, yf, flight_r.h.values]).reshape(3, -1).T
# dist = distance_matrix(point3d, point2d)
# dist_0 = np.where(dist > 25000 * aero.ft, -100, dist)

# noise = np.zeros(dist_0.shape)
# num = len(flight_r)
# for i in range(num):
#     thr = flight_r.thrust.values[i]
#     ns = interp_npd(np.array([np.array([thr] * len(dist[i])), dist[i]]).T)

#     noise[i] = np.where(dist_0[i] < 0, 0, ns)
# noise1 = noise.reshape(num, len(pop_y), -1)
# population = pop_map.pp.values.reshape(len(pop_y), len(pop_x))

# for treshold in noise_tresholds:
#     affected_pop = np.zeros(dist_0.shape)
#     for i in range(num):
#         affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

#     people_affected = (
#         np.where(
#             np.sum(affected_pop.reshape(num, len(pop_y), -1), axis=0) > 0,
#             population.reshape(len(pop_y), -1),
#             0,
#         )
#     ).sum()
#     pp_dict.append(
#         {
#             "fid": fid,
#             "people_affected": people_affected,
#             "pp_total": pp_total,
#             "map_type": map_type,
#             "max_fuel": "real",
#             "treshold": treshold,
#         }
#     )
# proj = ccrs.PlateCarree()
# proj = ccrs.TransverseMercator(
#     central_longitude=eham["lon"], central_latitude=eham["lat"]
# )
# trans = ccrs.PlateCarree()
# fig, ax = plt.subplots(
#     1,
#     1,
#     figsize=(6, 6),
#     subplot_kw=dict(projection=proj),
# )
# ax.set_extent([4, 5.5, 51.4, 52.65])

# ax.add_feature(BORDERS, linestyle="dotted", alpha=0.6)
# ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.6)
# ax.add_feature(LAKES, linestyle="dotted", alpha=0.6)
# ax.add_feature(RIVERS, linestyle="dotted", alpha=0.6)
# gl = ax.gridlines(draw_labels=False, color="gray", alpha=0.5, ls="--")
# gl.bottom_labels = True
# gl.left_labels = True
# gl.xlabel_style = {
#     "size": 6,
# }
# gl.ylabel_style = {
#     "size": 6,
# }

# ax.plot(
#     [flonmax, flonmax, flonmin, flonmin, flonmax],
#     [flatmin, flatmax, flatmax, flatmin, flatmin],
#     transform=trans,
#     c="k",
#     linestyle="dashed",
#     alpha=0.5,
# )

# pop_filtered = pop_map0.query("10 < pp")

# pp_alpha = (pop_filtered["pp"] - 10) / (20000 - 10)
# pp_alpha = np.clip(pp_alpha, 0, 1)  # clamp between 0 and 1

# colors = [mcolors.to_rgba("tab:blue", alpha=a) for a in pp_alpha]
# ax.scatter(
#     pop_filtered.lon,
#     pop_filtered.lat,
#     c=colors,
#     s=5,
#     marker="s",
#     transform=trans,
#     edgecolors="none",
# )

# norm = plt.Normalize(vmin=0, vmax=1, clip=True)
# pop_lon, pop_lat = transformer_ll.transform(X2d, Y2d)
# # noise45 = np.sum(noise1 > 45, axis=0)

# # cmap = ["hsv_r", "Purples_r", "Wistia_r", "Greens_r"]
# # colors = ["r", "purple", "darkorange", "darkgreen"]
# cmap = ["hsv_r", "Purples_r", "Greens_r"]
# colors = ["r", "purple", "darkgreen"]
# manual_positions = [
#     # 45: [(4.75, 51.2)],
#     [transformer_xy.transform(4.8, 51.8)],
#     [transformer_xy.transform(4.81, 52.0)],
#     [transformer_xy.transform(4.8, 52.2)],
# ]
# for i, db in enumerate(noise_tresholds):
#     # noise_n = gaussian_filter(noise_n, sigma=0.1)
#     noise_n = np.sum(noise1 > db, axis=0)
#     cs = ax.contour(
#         pop_lon[:, :, 0],
#         pop_lat[:, :, 0],
#         noise_n,
#         norm=norm,
#         cmap=cmap[i],
#         levels=0,
#         transform=trans,
#         linewidth=2,
#         alpha=1,
#         # label = db,
#     )
#     labels = ax.clabel(
#         cs,
#         fmt={0: f"{db}"},  # Force label to show the dB threshold
#         fontsize=10,
#         inline=True,
#         colors=colors[i],
#         # manual=manual_positions[i],
#     )
#     for label in labels:
#         label.set_fontweight("bold")
#         label.set_fontfamily("monospace")
# lonf, latf = transformer_ll.transform(xf, yf)

# ax.plot(
#     lonf,
#     latf,
#     lw=2,
#     c="k",
#     transform=trans,
#     alpha=0.5,
# )
# # ax.scatter(lonf,latf,s=1, c="k", transform=trans)
# norm = plt.Normalize(vmin=-10, vmax=100, clip=True)
# if max_fuel:
#     plt.savefig(
#         f"figs/contours_n_{map_type}_real_opt_18L.png",
#         pad_inches=0,
#         bbox_inches="tight",
#         dpi=500,
#     )
# else:
#     plt.savefig(
#         f"figs/contours_n_{map_type}_real_opt_18L.png",
#         pad_inches=0,
#         bbox_inches="tight",
#         dpi=500,
#     )
# plt.show()

# df_pp = pd.DataFrame.from_dict(pp_dict)

# df_pp = df_pp.assign(percentage=lambda x: x.people_affected / x.pp_total * 100)
# df_pp
