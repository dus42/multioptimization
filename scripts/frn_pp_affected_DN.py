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
import sys
import os

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath("../src/"))
from pp_affected import pp_affected, cost_grid_cost, get_npd_interpolator

# %%
ac = "a320"
fuelflow = FuelFlow(ac)
thrust = Thrust(ac)

curr_path = os.path.dirname(os.path.realpath(__file__))
options = itertools.product([10, 35, 45, 55, 65, 75], ["fuel", "real", "night", "day"])

pp_dict = []
for treshold, frnd in options:
    print(treshold, frnd)
    pop_map = pd.read_parquet(
        curr_path + f"/../data_generated/static_pop.parquet"
    ).rename(columns={"value": "pp", "x": "lon", "y": "lat"})
    # pop_map=pd.read_parquet(curr_path + f"/../data_raw/pop_static.parquet")
    # flights = pd.read_csv(f"../data_generated/optimal_flights_DN_39.csv")
    flights = pd.read_parquet(f"../data_generated/flight_params_frnd_DN.parquet")
    if frnd == "fuel":
        flights = flights.query("obj=='fuel'")
    elif frnd == "night":
        flights = flights.query("obj=='pop_night'")
    elif frnd == "day":
        flights = flights.query("obj=='pop_day'")
    else:
        flights = flights.query("obj=='real'")
    df_cost = pd.read_parquet(
        curr_path + f"/../data_generated/df_cost_DN_ext_18k.parquet"
    )

    for fid in flights.fid.unique():
        flight = flights.query("fid==@fid").reset_index(drop=True)
        grid_type = "ll"
        flight = cost_grid_cost(df_cost, flight)
        pp_aff_all, pp_aff_uni, flight = pp_affected(
            ac, pop_map, grid_type, flight, treshold
        )

        pp_dict.append(
            {
                "fid": fid,
                "fuel": flight.mass.max() - flight.mass.min(),
                "cost_grid": flight.cost_grid.sum(),
                "people_affected_sum_all": pp_aff_all,
                "pp_unique_sum": pp_aff_uni,
                "treshold": treshold,
                "frnd": frnd,
            }
        )

df = pd.DataFrame.from_dict(pp_dict)
df.to_csv(
    curr_path + f"/../data_generated/pp_affected_frnd_DN.csv",
    index=False,
)

# %%
import glob
import matplotlib.pyplot as plt
import seaborn as sns


map_type = "DN"
df = pd.read_csv(
    curr_path + f"/../data_generated/pp_affected_frnd_DN.csv",
)
fig, ax = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1, 1]})
df["order"] = df["frnd"].apply(
    lambda x: {"fuel": 2, "real": 3, "night": 0, "day": 1}[x]
)
df = df.sort_values(by="order")
g = sns.barplot(
    df.query("105>treshold>10"), x="treshold", y="pp_unique_sum", hue="frnd", ax=ax[0]
)
sns.barplot(
    df.query("treshold==35"),
    x="frnd",
    y="pp_unique_sum",
    hue="frnd",
    ax=ax[1],
    legend=None,
)
sns.barplot(
    df.query("treshold==35"), x="frnd", y="fuel", hue="frnd", ax=ax[2], legend=None
)
g.set_yscale("log")
ax[0].set_xlabel("Noise level, dB LAmax")
ax[0].set_ylabel("")
ax[0].legend(title=None)

ax[0].set_title("People affected per treshold, log-scale")
ax[1].set_title("People affected by noise\nlevel above 35 LAmax, dB")
ax[1].set_ylabel("")
ax[1].set_xlabel("")

ax[2].set_title("Fuel spent, kg")
ax[2].set_ylabel("")
ax[2].set_xlabel("")
ax[2].set_ylim(1200, 1600)

plt.tight_layout()
plt.savefig(
    curr_path + f"/../figs/DN_barchart_all4.png",
    bbox_inches="tight",
    dpi=300,
)
plt.show()


# %%LATEX table

df_35 = df[df["treshold"] == 35]

table = df_35.groupby("frnd", as_index=False).agg(
    {"pp_unique_sum": "mean", "fuel": "mean"}
)

print(table.to_latex(index=False))
# %%
import glob
import matplotlib.pyplot as plt
import seaborn as sns

map_type = "DN"

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
df["order"] = df["frnd"].apply(
    lambda x: {"fuel": 2, "real": 3, "night": 0, "day": 1}[x]
)
df = df.sort_values(by="order")
sns.barplot(df, x="treshold", y="pp_unique_sum", hue="frnd", ax=ax[0])
sns.barplot(
    df,
    x="fid",
    y="people_affected_sum_all",
    hue="frnd",
    ax=ax[1],
)

ax[0].set_xlabel("Noise level, dB LAmax")
ax[1].set_xlabel("Noise level, dB LAmax")
ax[0].legend(title=None)
ax[1].legend(title=None)
# ax[0].set_ylim(0, 230_000)
# ax[1].set_ylim(0, 400_000)
# plt.suptitle(op + " " + day_night)
# plt.tight_layout()
# plt.savefig(
#     f"{root_dir}/../figures/{op}_barchart_{day_night}.png",
#     bbox_inches="tight",
#     dpi=300,
# )
plt.show()
# %%
# flights = pd.read_csv(f"../data_generated/optimal_flights_DN_39.csv")
flights = pd.read_parquet(f"../data_generated/flight_params_frnd_DN.parquet")
fig, axes = plt.subplots(2, 2, figsize=(9, 9))

for fid in flights.fid.unique():
    for i, obj in enumerate(flights.obj.unique()):
        f = flights.query("fid==@fid and obj==@obj")
        axes.flatten()[i].plot(f.altitude, f.obj_T)
        axes.flatten()[i].set_title(obj)
        print(
            fid,
            f.altitude.max(),
            f.altitude.min(),
        )

# %%
f = flights.query("fid==@fid and obj=='real'")
# %%
f
# %%
