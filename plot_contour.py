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
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE, LAKES, RIVERS
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns
from traffic.visualize.markers import rotate_marker, aircraft
from pathlib import Path

root_dir = Path(__file__).resolve().parent
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plt_contour(pop_map0, noise, flight, pop_lat, pop_lon):
    pop_filtered = pop_map0.query("10 < pp")
    Z = np.max(noise1, axis=0)
    Z = noise.max(axis=0).reshape(len(pop_lat), len(pop_lon))

    # Z2 = affected_pop.reshape(len(flight), len(pop_y), -1)[:]
    # Create a 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    trans = ccrs.PlateCarree()
    zf = flight.iloc[:, :].altitude.to_numpy() * aero.ft
    fsc = flight.iloc[:, :].query("cost_grid==0").query("h==h.min()")
    if len(fsc) < 1:
        fsc = flight.iloc[:, :].query("h==h.min()")
    zsc = fsc.altitude.values * aero.ft
    xfsc, yfsc = transformer_xy.transform(fsc.longitude, fsc.latitude)
    xf, yf = transformer_xy.transform(
        flight.iloc[:, :].longitude, flight.iloc[:, :].latitude
    )
    z_values = range(0, 110, 8)
    if len(flight.query("cost_grid==0")) > 1:

        cut = flight.query("cost_grid==0").index[0]
        xh, yh, zh = xf[:cut], yf[:cut], zf[:cut]
        xt, yt, zt = xf[cut:], yf[cut:], zf[cut:]
    else:
        cut = flight.index[-1]
        xh, yh, zh = xf[:], yf[:], zf[:]
        xt, yt, zt = xf[:], yf[:], zf[:]

    norm = plt.Normalize(vmin=45, vmax=75, clip=True)
    cntr = ax.contourf(
        pop_lon,
        pop_lat,
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
        transform=trans,
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

    ax.set_proj_type("persp", focal_length=0.3)
    ax.view_init(elev=30, azim=-25, roll=0)
    Z2 = Z
    Z2 = np.zeros(Z.shape)

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

    zticks = ax.get_zticks()
    ax.set_xlim(pop_x.min(), pop_x.max())
    ax.set_ylim(pop_y.min(), pop_y.max())
    ax.set_zlim(zticks.min(), zticks.max())
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()

    ax.set_xticks([])
    ax.set_yticks([])

    ax.scatter3D(xh, yh, zh, s=1, c="tab:red", zorder=30)
    ax.plot3D(xh, yh, zh, c="tab:red", zorder=30)
    ax.scatter3D(xt, yt, zt, s=1, c="tab:blue", zorder=30)
    if op == "arr":
        rot = 90
    else:
        rot = 270
    ax.scatter(
        xfsc[-1],
        yfsc[-1],
        zsc[-1],
        marker=rotate_marker(aircraft, 90),
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

    ax.scatter3D(
        xfsc[0], yfsc[0], zsc[0], c="navy", s=20, marker="*", zorder=60, alpha=0.7
    )
    ax.scatter3D(xfsc[0], yfsc[0], 0, c="navy", s=40, marker="*", zorder=20, alpha=1)
    ax.scatter3D(
        xfsc[0], yticks.max(), zsc[0], c="navy", s=40, marker="*", zorder=20, alpha=0.4
    )
    ax.scatter3D(
        xticks.min(), yfsc[0], zsc[0], c="navy", s=40, marker="*", zorder=20, alpha=0.4
    )

    ax.set_zticks(np.arange(0, 35000 * aero.ft, 5000 * aero.ft))
    zticks = ax.get_zticks()
    ax.set_zticklabels((np.round(zticks / aero.ft / 1000, 0)).astype(int))

    ax.set_xlabel("Longitude", labelpad=-10, fontsize=10)
    ax.set_ylabel("Latitude", labelpad=-15, fontsize=10)
    ax.set_zlabel("Altitude, kft")
    ax.set_title("cluster " + str(flight.cluster.iloc[0]))

    plt.tight_layout()
    plt.savefig(
        f"{root_dir}/../figures/{op}_3dplot_cl{flight.cluster.iloc[0]}_{day_night}.png",
        pad_inches=0.3,
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


# %%
_npdc = None


def _get_npdc():
    global _npdc
    if _npdc is None:
        npdc = pd.read_csv(f"data_raw/npds_check.csv").query("noise_metric=='LAmax'")
        thrusts = npdc.thrust_n.to_numpy()
        dists = np.array([int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]])
        noise_levels = npdc.iloc[:, 5:15].values
        _npdc = (thrusts, dists, noise_levels)
    return _npdc


def interp_npd1(thrs, dis):
    thrusts, dists, noise_levels = _get_npdc()
    log_dists = np.log10(dists)  # precompute once

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


# %%
def plt_contour(
    pop_map0,
    noise,
    flight,
    pop_lat,
    pop_lon,
    *,
    do_3d=True,
    do_2d=True,
    levels=np.arange(35, 76, 10),
    vmin=35,
    vmax=65,
    save_dir=None,  # e.g. root_dir / "figs"
    tag=None,  # e.g. f"{flight.fid.iloc[0]}_{flight.obj.iloc[0]}"
    to_scale=False,
):
    """
    Drop-in replacement for the call at line 413:
      plt_contour(pop_map0, noise, flight, pop_lat, pop_lon)
    Produces:
      - 3D ground contour (in EPSG:3035 meters) + trajectory
      - 2D Cartopy contour (in lon/lat degrees) + trajectory
    """
    # ---- Validate / shape ground field ----
    Z = noise.max(axis=0).reshape(len(pop_lat), len(pop_lon))  # (lat, lon)
    # Build lon/lat mesh
    Lon2d, Lat2d = np.meshgrid(pop_lon, pop_lat, indexing="xy")
    # Use the module-global transformer if available; else make one
    try:
        _transformer_xy = transformer_xy  # noqa: F821
    except NameError:
        _transformer_xy = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(3035), always_xy=True
        )
    # Convert grid to EPSG:3035 for 3D plotting
    X2d, Y2d = _transformer_xy.transform(Lon2d, Lat2d)

    # Population points (optional overlay)
    ymin, ymax = (3100000, 3300000)
    xmin, xmax = (3800000, 4020000)
    pop_map0 = pop_map0.query("@xmin<x<@xmax and @ymin<y<@ymax")
    pop_pts = pop_map0
    if all(c in pop_map0.columns for c in ("longitude", "latitude")):
        lon_min, lon_max = float(np.min(pop_lon)), float(np.max(pop_lon))
        lat_min, lat_max = float(np.min(pop_lat)), float(np.max(pop_lat))
        pop_pts = pop_map0.query(
            "@lon_min <= longitude <= @lon_max and @lat_min <= latitude <= @lat_max"
        )
    if "pp" in pop_pts.columns:
        pop_pts = pop_pts.query("pp > 1500")
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # ---- 3D plot (no Cartopy transforms on Axes3D) ----
    # Flight trajectory in EPSG:3035 + altitude
    flight = flight.query(
        f"{pop_map0.longitude.min()}<longitude<{pop_map0.longitude.max()} and {pop_map0.latitude.min()}<latitude<{pop_map0.latitude.max()}"
    ).iloc[:-5, :]
    flon = flight.longitude.to_numpy()
    flat = flight.latitude.to_numpy()
    xf, yf = _transformer_xy.transform(flon, flat)
    if "h" in flight.columns:
        zf = flight.h.to_numpy() / 1000 / aero.ft
    elif "altitude" in flight.columns:
        zf = flight.altitude.to_numpy() / 1000
    else:
        zf = np.zeros_like(xf)
    # Split "highlight" vs "tail" (optional)
    xh, yh, zh = xf, yf, zf
    xt, yt, zt = np.array([]), np.array([]), np.array([])
    if "cost_grid" in flight.columns and (flight["pp"] < 1).any():
        cut = int(flight.index[flight["pp"] < 1][0])
        xh, yh, zh = xf[:cut], yf[:cut], zf[:cut]
        xt, yt, zt = xf[cut:], yf[cut:], zf[cut:]
    if do_3d:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        cntr = ax.contourf(
            X2d,
            Y2d,
            Z,
            levels=levels,
            zdir="z",
            offset=0,
            cmap="plasma",
            alpha=1.0,
            norm=norm,
        )
        # Population dots on ground (needs x/y in EPSG:3035)
        if all(c in pop_pts.columns for c in ("x", "y")):
            ax.scatter3D(
                pop_pts["x"].to_numpy(),
                pop_pts["y"].to_numpy(),
                0,
                s=0.1,
                c=pop_pts["pp"].to_numpy(),
                cmap="cividis",
                alpha=0.25,
                zorder=10,
            )
        if len(xh) > 1:
            ax.plot3D(xh, yh, zh, c="tab:red", lw=2, zorder=30)
            ax.scatter3D(xh, yh, zh, s=2, c="tab:red", zorder=31)
            ax.plot3D(
                xh,
                np.zeros(len(xh)) + ymax,
                zh,
                c="tab:red",
                lw=1.5,
                zorder=1,
                alpha=0.3,
            )
            if to_scale:
                ax.plot3D(
                    np.zeros(len(xh)) + xmin,
                    yh,
                    zh,
                    c="tab:red",
                    lw=1.5,
                    zorder=1,
                    alpha=0.3,
                )
        if len(xt) > 0:
            ax.scatter3D(xt, yt, zt, s=2, c="tab:blue", zorder=32)
            ax.plot3D(
                xt,
                np.zeros(len(xt)) + ymax,
                zt,
                c="tab:blue",
                lw=1.5,
                zorder=1,
                alpha=0.3,
            )
            if to_scale:
                ax.plot3D(
                    np.zeros(len(xt)) + xmin,
                    yt,
                    zt,
                    c="tab:blue",
                    lw=1.5,
                    zorder=1,
                    alpha=0.3,
                )

        ax.plot3D(
            xf, yf, np.zeros(len(flight)), c="k", linestyle="dashed", lw=1, zorder=1
        )

        for i in np.arange(0, len(flight), 5):
            ax.plot3D(
                [xf[i], xf[i]],
                [yf[i], yf[i]],
                [zf[i], 0],
                c="k",
                lw=1,
                zorder=33,
                alpha=0.2,
            )
            ax.plot3D(
                [xf[i], xf[i]],
                [yf[i], ymax],
                [zf[i], zf[i]],
                c="k",
                lw=1,
                zorder=33,
                alpha=0.05,
            )
            if to_scale:
                ax.plot3D(
                    [xf[i], xmin],
                    [yf[i], yf[i]],
                    [zf[i], zf[i]],
                    c="k",
                    lw=1,
                    zorder=33,
                    alpha=0.05,
                )

        ax.set_proj_type("persp", focal_length=0.3)
        if not to_scale:
            ax.view_init(elev=30, azim=-65, roll=0)
            ax.set_xlabel("x (EPSG:3035)", labelpad=-15)
            ax.set_ylabel("y (EPSG:3035)", labelpad=-15)
            cbar = plt.colorbar(cntr, shrink=0.4, orientation="horizontal", pad=0.01)
            ax.set_zlabel("Altitude, kft")
        else:

            x_range = xmax - xmin
            y_range = ymax - ymin
            z_range = (zf.max() - zf.min()) * 1000 * aero.ft  # in kft now
            # z_scale = (x_range / z_range) * 0.15
            ax.set_box_aspect([x_range, y_range, z_range * 1])
            ax.view_init(elev=10, azim=-35, roll=0)
            ax.set_zticks([0, 40])
            ax.set_xlabel("x (EPSG:3035)", labelpad=-10)
            ax.set_ylabel("y (EPSG:3035)")
            cbar = plt.colorbar(cntr, shrink=0.4, orientation="horizontal", pad=-0.3)
            ax.set_zlabel("Altitude, kft", labelpad=-3)
            if tag is not None:
                tag = tag + "_toscale"
            else:
                tag = "_toscale"

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.set_zlim(0, 40)

        cbar.set_ticks(np.arange(vmin, vmax + 0.1, 10))
        cbar.ax.set_xlabel(r"$L_{A_{max}}$ dBA", rotation=0, labelpad=5)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        if tag is None and "fid" in flight.columns:
            tag = str(flight["fid"].iloc[0])
        if save_dir is not None:
            out = Path(save_dir)
            name = f"contour_3d{'' if tag is None else '_' + tag}.png"
            fig.savefig(out / name, bbox_inches="tight", pad_inches=0.2, dpi=300)
            if to_scale:
                fig.savefig(out / name, bbox_inches="tight", pad_inches=0.6, dpi=300)
        plt.show()
    # ---- 2D plot (Cartopy, lon/lat) ----
    if do_2d:
        proj = ccrs.EuroPP()
        fig, ax = plt.subplots(1, 1, figsize=(4, 6), subplot_kw=dict(projection=proj))
        ax.add_feature(BORDERS, linestyle="dotted", alpha=0.6)
        ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.6)
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        # gl.bottom_labels = True
        # gl.right_labels = True

        gl.ylocator = mticker.FixedLocator([51.5, 52, 52.5])
        gl.xlocator = mticker.FixedLocator([4, 4.5, 5, 5.5])

        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
        ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)

        # ax.set_facecolor('lightgray')
        ax.set_extent(
            [
                float(np.min(pop_lon)) - 0.1,
                float(np.max(pop_lon)) - 0.2,
                float(np.min(pop_lat)),
                float(np.max(pop_lat)),
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_extent([4, 5.5, 51.4, 52.65])
        cs = ax.contour(
            pop_lon,
            pop_lat,
            Z,
            levels=levels,
            cmap="plasma",
            norm=norm,
            transform=ccrs.PlateCarree(),
            alpha=0.9,
        )
        ax.clabel(cs, fontsize=12)
        ax.plot(
            flon,
            flat,
            lw=2,
            c="k",
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )
        if all(c in pop_pts.columns for c in ("longitude", "latitude")):
            ax.scatter(
                pop_pts["longitude"].to_numpy(),
                pop_pts["latitude"].to_numpy(),
                s=5,
                c=pop_pts["pp"].to_numpy(),
                cmap="cividis",
                alpha=0.15,
                transform=ccrs.PlateCarree(),
            )
        if tag is None and "fid" in flight.columns:
            tag = str(flight["fid"].iloc[0])
        if save_dir is not None:
            out = Path(save_dir)
            name = f"contour_2d{'' if tag is None else '_' + tag}.png"
            fig.savefig(out / name, bbox_inches="tight", dpi=300)
        plt.show()


# %%
# options = itertools.product(["D", "N"], [True, False])
# options = itertools.product(["D"], [False])
plot = True
# options = itertools.product( ["f", "r", "d","n"])
df_cost = pd.read_parquet(f"data_generated/df_cost_DN_ext.parquet")
# options = pd.DataFrame(options, columns=["frdn"])
crs_3035 = CRS.from_epsg(3035)
crs_4326 = CRS.from_epsg(4326)
transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)
eham = nav.airport("EHAM")
noise_tresholds = [45, 55, 65, 75, 85]
flights = pd.read_csv(
    f"data_generated/optimal_flights_DN_99.csv",
)
pp_dict = []
# for i, frdn in enumerate(["fuel", "real", "pop_day", "pop_night"]):
for i, frdn in enumerate(["pop_night"]):
    # fid = 'VLG28E_184'
    df_real0 = flights.query("obj=='real'")
    max_lola = (df_real0.longitude.max() + 0.3, df_real0.latitude.max() + 0.3)
    min_lola = (df_real0.longitude.min() - 0.3, df_real0.latitude.min() - 0.3)
    min_x, min_y = transformer_xy.transform(*min_lola)
    max_x, max_y = transformer_xy.transform(*max_lola)
    fids = df_real0.fid.unique()
    fids = ["VLG53KX_284"]
    for fid in tqdm(fids[:], desc=frdn):
        # if fid in ["AFR1500_101", "DLH4NV_123", "BAW450Q_115"]:
        #     continue
        df_real = df_real0.query("fid==@fid").reset_index(drop=True)

        # pop_map0 = pd.read_parquet(f"{root_dir}/../data/maps/pop_static.parquet")
        # pop_map0["x"], pop_map0["y"] = transformer_xy.transform(
        #     pop_map0["lon"], pop_map0["lat"]
        # )
        pop_map0 = pd.read_parquet(f"data_generated/static_pop.parquet").rename(
            columns={"x": "longitude", "y": "latitude", "value": "pp"}
        )
        x, y = transformer_xy.transform(pop_map0.longitude, pop_map0.latitude)
        pop_map0 = pop_map0.assign(x=x, y=y)
        pop_map = (
            pop_map0.query(
                f"{min_lola[0]}<longitude<{max_lola[0]} and {min_lola[1]}<latitude<{max_lola[1]}"
            )
            .fillna(0)
            .copy()
        )
        pop_lat = np.sort(pop_map.latitude.unique())
        pop_lon = np.sort(pop_map.longitude.unique())
        grid = (
            pop_map.pivot(index="latitude", columns="longitude", values="pp")
            .reindex(index=pop_lat, columns=pop_lon, fill_value=0.0)
            .fillna(0.0)
        )
        Lon2d, Lat2d = np.meshgrid(pop_lon, pop_lat, indexing="xy")
        X2d, Y2d = transformer_xy.transform(Lon2d, Lat2d)
        point2d = np.array([X2d, Y2d, np.zeros_like(X2d)]).reshape(3, -1).T
        population = grid.to_numpy().ravel()
        if population.shape[0] != point2d.shape[0]:
            raise ValueError(
                f"Population/point grid mismatch: population={population.shape[0]} vs point2d={point2d.shape[0]}"
            )
        npdc = pd.read_csv(f"data_raw/npds_check.csv").query("noise_metric=='LAmax'")
        thrusts = npdc.thrust_n.to_numpy()
        dists = [int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]]
        noise_levels = npdc.iloc[:, 5:15].values
        # interp_npd = RegularGridInterpolator(
        #     (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
        # )

        flight = flights.query("fid==@fid").query("obj==@frdn").reset_index(drop=True)
        # flight = assign_thrust_df(flight)

        [flonmin, flonmax, flatmin, flatmax] = [
            flight.longitude.min() - 0.1,
            flight.longitude.max() + 0.1,
            flight.latitude.min() - 0.1,
            flight.latitude.max() + 0.1,
        ]
        pp_total = pop_map.query(
            f"{flonmin}<longitude<{flonmax} and {flatmin}<latitude<{flatmax}"
        ).pp.sum()
        xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
        point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
        dist = distance_matrix(point3d, point2d)
        dist_0 = np.where(dist > 25000 * aero.ft, -100, dist)

        noise = np.zeros(dist_0.shape)
        num = len(flight)
        for i in range(num):
            thr = flight.thrust.values[i]
            delta = aero.pressure(flight.h.values[i]) / aero.pressure(0)
            cnt_per_eng = thr / delta / 2
            ns = interp_npd1(np.full(dist_0.shape[1], cnt_per_eng), dist_0[i])

            noise[i] = np.where(dist_0[i] < 0, 0, ns)
        # aggregated_noise = np.sum(noise.reshape(num, len(pop_y), -1), axis=0)
        noise1 = noise.reshape(num, len(pop_lat), -1)

        for treshold in noise_tresholds:
            affected_pop = np.zeros(dist_0.shape)
            for i in range(num):
                affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

            # agg_pop = np.sum(affected_pop.reshape(num, len(pop_y), -1), axis=0)
            # people_affected = sum(sum(agg_pop))
            people_affected = (
                np.where(
                    np.sum(affected_pop.reshape(num, len(pop_lat), -1), axis=0) > 0,
                    population.reshape(len(pop_lat), -1),
                    0,
                )
            ).sum()
            pp_dict.append(
                {
                    "fid": fid,
                    # "op": op,
                    # "cluster": flight.cluster.iloc[0],
                    "frdn": frdn,
                    "people_affected": people_affected,
                    "pp_total": pp_total,
                    # "day_night": day_night,
                    "treshold": treshold,
                }
            )
            if treshold == 45:
                flight = flight.assign(pp=affected_pop.sum(axis=1))

        if plot:
            plt_contour(pop_map0, noise, flight, pop_lat, pop_lon, save_dir="figs/")

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
        # # ax.set_extent([4, 5.5, 51.4, 52.65])

        # ax.add_feature(BORDERS, linestyle="dotted", alpha=0.6)
        # ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.6)
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
        # airports["EHAM"].runways["18L"].plot(ax=ax, alpha=0.5)

        # pop_filtered = pop_map.query("10 < pp")

        # # Normalize pp to [0, 1] with 10 mapped to 0, 2000+ mapped to 1
        # pp_alpha = (pop_filtered["pp"] - 10) / (20000 - 10)
        # pp_alpha = np.clip(pp_alpha, 0, 1)  # clamp between 0 and 1

        # # Set constant color (black) and variable alpha
        # colors = [mcolors.to_rgba("tab:blue", alpha=a) for a in pp_alpha]
        # # ax.scatter(
        # #     pop_filtered.lon,
        # #     pop_filtered.lat,
        # #     c=colors,
        # #     s=5,
        # #     marker="s",
        # #     transform=trans,
        # #     edgecolors="none",
        # # )

        # cmap = plt.get_cmap("CMRmap_r")
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        #     "narrow_gnuplot", cmap(np.linspace(0.5, 1, 256))
        # )

        # pop_lon, pop_lat = transformer_ll.transform(X2d, Y2d)

        # noise_n = np.max(noise1, axis=0)
        # norm = plt.Normalize(vmin=55, vmax=70, clip=True)
        # cs = ax.contour(
        #     pop_lon[:, :, 0],
        #     pop_lat[:, :, 0],
        #     noise_n,
        #     norm=norm,
        #     cmap=cmap,
        #     levels=[55, 65, 75],
        #     transform=trans,
        #     linewidth=2,
        #     alpha=1,
        # )
        # labels = ax.clabel(
        #     cs,
        #     fontsize=10,
        # )
        # for label in labels:
        #     label.set_fontweight("bold")
        #     label.set_fontfamily("monospace")

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

        # # plt.savefig(
        # #     f"{root_dir}/../figures/contours_n_{day_night}.png",
        # #     pad_inches=0,
        # #     bbox_inches="tight",
        # #     dpi=500,
        # # )
        # plt.show()

df_pp = pd.DataFrame.from_dict(pp_dict)
# pp_total = pop_map.query(
#     f"{flonmin}<lon<{flonmax} and {flatmin} < lat<{flatmax}"
# ).pp.sum()
# df_pp = df_pp.assign(pp_total=pp_total)
df_pp = df_pp.assign(percentage=lambda x: x.people_affected / x.pp_total * 100)
df_pp.to_csv("df_pp_contrours.csv", index=False)


# %%%
def nd(val):
    if val == "N":
        return "Night"
    else:
        return "Day"


df = df_pp.sort_values(by=["frdn", "percentage"], ascending=False)
df["treshold"] = df["treshold"].apply(lambda x: str(x) + " dBA")
# df["percentage"] = df["percentage"].apply(lambda x: str(round(x,2))+" %")
# df["day_night"] = df["day_night"].apply(nd)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(
    df,
    x="treshold",
    y="people_affected",
    hue="frdn",
    ax=ax,
)
# ax.set_yscale("log")
ax.set_xlabel("")
ax.set_ylabel("People affected %")
ax.legend(title=None)


# %%


def perc(val):
    return f"{val:.3f}~\\%"


def db(val):
    return f"{val}~dBA"


df_small = df_pp.pivot_table(
    index=["treshold"], columns="day_night", values="percentage"
).reset_index()
df_small["D"] = df_small["D"].apply(perc)
df_small["N"] = df_small["N"].apply(perc)
df_small["treshold"] = df_small["treshold"].apply(db)
df_small.columns.name = None
df_small.rename(columns={"D": "Day", "N": "Night"}, inplace=True)

df_small[["Day", "Night"]] = df_small[["Day", "Night"]].round(2)


print(df_small[["treshold", "Day", "Night"]].to_latex(index=False))
# %%
dt = flight.ts.diff().fillna(0).to_numpy()
Z = np.max(noise1, axis=0)
Z = noise.max(axis=0).reshape(len(pop_y), len(pop_x))
# 3) Coordinates in EPSG:3035 (meters) + altitude (meters)
xf, yf = transformer_xy.transform(
    flight.longitude.to_numpy(), flight.latitude.to_numpy()
)
zf = flight.altitude.to_numpy() * aero.ft
# Split trajectory
if op == "arr":
    cut = flight.query("grid_cost==0").index[-1]
    xh, yh, zh = xf[cut:], yf[cut:], zf[cut:]
    xt, yt, zt = xf[:cut], yf[:cut], zf[:cut]
else:
    cut = flight.query("grid_cost==0").index[0]
    xh, yh, zh = xf[:cut], yf[:cut], zf[:cut]
    xt, yt, zt = xf[cut:], yf[cut:], zf[cut:]

# Optional: population points to show on ground plane
pop_pts = pop_map.query(
    "@pop_x.min()<=x<=@pop_x.max() and @pop_y.min()<=y<=@pop_y.max()"
)
pop_pts = pop_pts.query("pp > 2000")  # tune threshold
# --- Plot ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
# x_all = np.concatenate([xf])
# y_all = np.concatenate([yf])
# z_all = np.concatenate([zf])
# x_min, x_max = float(x_all.min()), float(x_all.max())
# y_min, y_max = float(y_all.min()), float(y_all.max())
# z_min, z_max = float(z_all.min()), float(z_all.max())
# max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
# x_mid = 0.5 * (x_min + x_max)
# y_mid = 0.5 * (y_min + y_max)
# z_mid = 0.5 * (z_min + z_min + (z_max - z_min)) / 2  # same as 0.5*(z_min+z_max)
# ax.set_xlim(x_mid - 0.5 * max_range, x_mid + 0.5 * max_range)
# ax.set_ylim(y_mid - 0.5 * max_range, y_mid + 0.5 * max_range)
# ax.set_zlim(0,  max_range)
# ax.set_box_aspect((1, 1, 1))
# Ground contour of max noise (Z must be shape (len(pop_y), len(pop_x)))
norm = plt.Normalize(vmin=45, vmax=75, clip=True)
cntr = ax.contourf(
    pop_x,
    pop_y,
    Z,
    levels=np.arange(35, 95, 10),
    zdir="z",
    offset=0,
    cmap="viridis",
    alpha=1.0,
    norm=norm,
)
# Population dots on the ground
ax.scatter3D(
    pop_pts.x.to_numpy(), pop_pts.y.to_numpy(), 0, s=0.1, c="k", alpha=0.35, zorder=10
)
# Trajectory: red line (impactful), blue scatter (negligible tail)
if cut > 1:
    ax.plot3D(xh, yh, zh, c="tab:red", lw=2, zorder=30)
    ax.scatter3D(xh, yh, zh, s=2, c="tab:red", zorder=31)
if len(xt) > 0:
    ax.scatter3D(xt, yt, zt, s=2, c="tab:blue", zorder=30)
# Cosmetics
ax.set_proj_type("persp", focal_length=0.3)
ax.view_init(elev=30, azim=-25, roll=0)
cbar = plt.colorbar(cntr, shrink=0.4, orientation="horizontal", pad=0.02)
cbar.set_ticks(np.arange(45, 85.0, 10))
cbar.ax.set_xlabel(r"$L_{A_{max}}$ dBA", rotation=0, labelpad=5)
ax.set_xlabel("x (EPSG:3035, m)")
ax.set_ylabel("y (EPSG:3035, m)")
ax.set_zlabel("Altitude (m)")

plt.tight_layout()
# plt.savefig(f"{root_dir}/../figures/3dplot.png", bbox_inches="tight", dpi=300)
# plt.show()


# %%
# df_cost = pd.read_csv(f"data_generated/df_cost_{day_night}.csv")
# df_real = (
#     pd.read_parquet(f"data_generated/arrs_centroids_D.parquet")
#     .query("flight_id=='AFR19LJ_067'")
#     .reset_index(drop=True)
# )
max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
min_x, min_y = transformer_xy.transform(*min_lola)
max_x, max_y = transformer_xy.transform(*max_lola)

# pop_map0 = pd.read_parquet(f"data_raw/{day_night}032011_1K_cropped.parquet").rename(
#     columns={"popul": "pp"}
# )

# pop_map = pop_map0.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
# pop_x = pop_map.x.unique()
# pop_y = pop_map.y.unique()
X2d, Y2d, Z2d = np.meshgrid(pop_x, pop_y, [0])
point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T


# npdc = pd.read_csv("npd_check.csv").query("noise_metric=='LAmax'")
thrusts = npdc.thrust_n.to_numpy()
dists = [int(c[2:-2]) * aero.ft for c in npdc.columns[5:15]]
noise_levels = npdc.iloc[:, 5:15].values
interp_npd = RegularGridInterpolator(
    (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
)

################################################################
#######################################

flight_r = df_real  # t[0].data.assign(h=lambda x: x.altitude * aero.ft)

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
            "day_night": "real",
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

pop_filtered = pop_map.query("10 < pp")

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

plt.savefig(
    f"{root_dir}/../figures/contours_n_{day_night}_real.png",
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


df = df_pp
df["order"] = df["day_night"].apply(order)
df["treshold"] = df["treshold"].apply(lambda x: str(x) + " dBA")
# df["percentage"] = df["percentage"].apply(lambda x: str(round(x,2))+" %")
df["day_night"] = df["day_night"].apply(nd)
df = df.sort_values(by="order")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(df, x="treshold", y="percentage", hue="day_night", ax=ax, errorbar=None)
ax.set_xlabel("")
ax.set_ylabel("People affected %")
ax.legend(title=None)
# ax.grid(True)
plt.savefig(
    f"{root_dir}/../figures/barchart_DN.png",
    pad_inches=0,
    bbox_inches="tight",
    dpi=500,
)


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


df = df_pp
df["order"] = df["day_night"].apply(order)
df["treshold"] = df["treshold"].apply(lambda x: str(x) + " dBA")
# df["percentage"] = df["percentage"].apply(lambda x: str(round(x,2))+" %")
df["day_night"] = df["day_night"].apply(nd)
df = df.sort_values(by="order")
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.barplot(
    df,
    x="treshold",
    y="percentage",
    hue="day_night",
    ax=ax,
    errorbar=None,
    palette=palette,
)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("")
ax.set_ylabel("People affected %")
ax.legend(title=None)
plt.tight_layout()
# plt.savefig(f"figs/barchart_DN.png", bbox_inches="tight", dpi=300)
plt.show()


# %%
def perc(val):
    return f"{val:.3f}~\\%"


def db(val):
    return f"{val}~dBA"


df_small = df_pp.pivot_table(
    index=["treshold"], columns="day_night", values="percentage"
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
