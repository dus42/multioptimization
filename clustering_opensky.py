# %%
from traffic.core import Traffic, Flight
from traffic.data import airports
from openap import Thrust, FuelFlow, aero, prop
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# %%
idd = "N"
if idd == "DN":
    t = Traffic(
        pd.read_parquet(
            "data_generated/opensky_flights_2024dn_ready_for_clustering.parquet"
        )
    )
else:
    t = Traffic(
        pd.read_parquet(
            "data_generated/opensky_flights_2024dn_ready_for_clustering.parquet"
        )
    ).query(f"day_night=='{idd}'")
ids = []
for f in t:
    if len(f) != 40:
        ids.append(f.flight_id)
t = t.query(f"flight_id not in {ids}").assign(h = lambda x:x.altitude*aero.ft)
# pd.DataFrame(list(t.data.callsign.unique()), columns=["callsign"]).to_csv(
#     "callsigns.csv", index=False
# )
# st = np.stack(list(f.data[["latitude", "longitude", "fl"]].values.ravel() for f in t))
# %%
t_dbscan = t.clustering(
    nb_samples=40,
    # projection=lcc(),
    features=["x", "y", "h"],
    clustering=DBSCAN(eps=1.2, min_samples=5),
    transform=StandardScaler(),
).fit_predict()
n_clusters = t_dbscan.data.cluster.max() + 1
print("Estimated number of clusters: %d" % n_clusters)
print("Estimated number of noise points: %d" % len(t_dbscan.data.query("cluster==-1")))
print("Estimated number of noise fligths: %d" % (len(t_dbscan.data.query("cluster==-1"))/40)
)
# t_dbscan = t_dbscan.query("cluster!=-1")
t_dbscan.data.cluster.hist()
# %%
saved_indices = []
for cluster in range(n_clusters):

    current_cluster = t_dbscan.query(f"cluster == {cluster}")
    ids = list(f.flight_id for f in current_cluster)
    indexx = current_cluster.centroid(40).flight_id  # Not sure about this centroid search, "copied" and adapted from traffic's git repo
    saved_indices.append(indexx)
# %%

colors = list(mcolors.TABLEAU_COLORS.keys())
colors = np.array(list(mcolors.CSS4_COLORS.keys()))
colors = colors[range(1, 148, 5)]
# colors.extend(["r", "b", "g", "y", "m", "c"])


for inx in saved_indices:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(6, 6))
    df = t_dbscan[inx].data
    for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
        ax0.plot(
            f.data.longitude,
            f.data.altitude,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax0.plot(
        df.longitude,
        df.altitude,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax0.plot(
        df.longitude,
        df.altitude,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax0.plot(
        df.longitude,
        df.altitude,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax0.set_title(f"cluster {df.cluster.values[0]}, {len(t_dbscan.query(f"cluster=={df.cluster.values[0]}"))}")
    for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
        ax1.plot(
            f.data.ts,
            f.data.vertical_rate,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax1.plot(
        df.ts,
        df.vertical_rate,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax1.plot(
        df.ts,
        df.vertical_rate,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax1.plot(
        df.ts,
        df.vertical_rate,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax1.set_title("vertical_rate")
    for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
        ax2.plot(
            f.data.ts,
            f.data.groundspeed,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax2.plot(
        df.ts,
        df.groundspeed,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax2.plot(
        df.ts,
        df.groundspeed,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax2.plot(
        df.ts,
        df.groundspeed,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax2.set_title("groundspeed")
    for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
        ax3.plot(
            f.data.ts,
            f.data.thrust,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax3.plot(
        df.ts,
        df.thrust,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax3.plot(
        df.ts,
        df.thrust,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax3.plot(
        df.ts,
        df.thrust,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax3.set_title("thrust")
    plt.show()

#%%
if idd == "DN":
    t_dbscan = t_dbscan.query("cluster not in [4,9,10,23]") ### DN
elif idd == "D":
    t_dbscan = t_dbscan.query("cluster not in [15,19]") ### D
else:
    t_dbscan = t_dbscan.query("cluster not in [5]") ### N
t = Traffic(t_dbscan.data.drop(columns=["cluster"]))
########################################################################################
# %%
t_dbscan_1 = t.clustering(
    nb_samples=40,
    # projection=lcc(),
    features=["x", "y", "altitude"],
    clustering=DBSCAN(eps=1.2, min_samples=8),
    transform=StandardScaler(),
).fit_predict()
n_clusters = t_dbscan_1.data.cluster.max() + 1
print("Estimated number of clusters: %d" % n_clusters)
print("Estimated number of noise points: %d" % len(t_dbscan_1.data.query("cluster==-1")))
print("Estimated number of noise fligths: %d" % (len(t_dbscan_1.data.query("cluster==-1"))/40)
)
t_dbscan_1 = t_dbscan_1.query("cluster!=-1")
t_dbscan_1.data.cluster.hist()
# %%
saved_indices = []
for cluster in range(n_clusters):

    current_cluster = t_dbscan_1.query(f"cluster == {cluster}")
    ids = list(f.flight_id for f in current_cluster)
    indexx = current_cluster.centroid(40).flight_id  # Not sure about this centroid search, "copied" and adapted from traffic's git repo
    saved_indices.append(indexx)
# %%

colors = list(mcolors.TABLEAU_COLORS.keys())
colors = np.array(list(mcolors.CSS4_COLORS.keys()))
colors = colors[range(1, 148, 5)]
# colors.extend(["r", "b", "g", "y", "m", "c"])


for inx in saved_indices:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(6, 6))
    df = t_dbscan_1[inx].data
    for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
        ax0.plot(
            f.data.longitude,
            f.data.altitude,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax0.plot(
        df.longitude,
        df.altitude,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax0.plot(
        df.longitude,
        df.altitude,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax0.plot(
        df.longitude,
        df.altitude,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax0.set_title(f"cluster {df.cluster.values[0]}, {len(t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"))}")
    for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
        ax1.plot(
            f.data.ts,
            f.data.vertical_rate,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax1.plot(
        df.ts,
        df.vertical_rate,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax1.plot(
        df.ts,
        df.vertical_rate,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax1.plot(
        df.ts,
        df.vertical_rate,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax1.set_title("vertical_rate")
    for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
        ax2.plot(
            f.data.ts,
            f.data.groundspeed,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax2.plot(
        df.ts,
        df.groundspeed,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax2.plot(
        df.ts,
        df.groundspeed,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax2.plot(
        df.ts,
        df.groundspeed,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax2.set_title("groundspeed")
    for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
        ax3.plot(
            f.data.ts,
            f.data.thrust,
            color=colors[f.data.cluster.values[0]],
            lw=2,
            linestyle="dashed",
            alpha=0.3,
        )
    ax3.plot(
        df.ts,
        df.thrust,
        color="w",
        lw=3.5,
        # linestyle="dashed",
        alpha=1,
    )
    ax3.plot(
        df.ts,
        df.thrust,
        color="k",
        lw=3,
        # linestyle="dashed",
        alpha=1,
    )
    ax3.plot(
        df.ts,
        df.thrust,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )
    ax3.set_title("thrust")
    plt.show()







#%%
ends = []
t_cnts = t_dbscan_1.query(f"flight_id in {saved_indices}")
for f in t_cnts:
    ends.append(
        [
            f.data.flight_id.values[0],
            f.data.latitude.values[-1],
            f.data.longitude.values[-1],
            f.data.tow.values[-1],
            f.data.altitude.values[-1],
            f.data.runway.values[-1],
        ]
    )
ends = pd.DataFrame(
    ends, columns=["fid", "latitude", "longitude", "tow", "altitude", "runway"]
)

ends.to_csv(f"data_generated/opensky_centroid_ends_{idd}.csv", index=False)
t_cnts.to_parquet(f"data_generated/opensky2024_centroids_{idd}.parquet", index=False)
t_dbscan_1.to_parquet(
    f"data_generated/opensky2024_clustered_flights_{idd}.parquet", index=False
)
# %%
colors = list(mcolors.TABLEAU_COLORS.keys())
colors.extend(["r", "b", "g", "y", "m", "c"])
ax = plt.subplot()

for f in t_dbscan_1:
    ax.plot(
        f.data.longitude,
        f.data.latitude,
        color=colors[f.data.cluster.values[0]],
        lw=2,
        linestyle="dashed",
        alpha=0.1,
    )
for inx in saved_indices:
    df = t_dbscan_1[str(inx)].data
    ax.plot(
        df.longitude,
        df.latitude,
        color=colors[df.cluster.values[0]],
        lw=2,
        # linestyle="dashed",
        alpha=1,
    )

# %%

t_dbscan_1 = Traffic(pd.read_parquet(
    f"data_generated/opensky2024_clustererd_flights_{idd}.parquet"))
ends=pd.read_csv(f"data_generated/opensky_centroid_ends_{idd}.csv")
table = []
run_dict= {"idd":1}
for inx in ends.fid:
    df_centr = t_dbscan_1[str(inx)].data
    t = t_dbscan_1.query(f"cluster=={df_centr.cluster.values[0]}")
    runways = t.data.runway.unique()
    rd = []
    for rwy in runways:
        nrf = len(t.query(f"runway == '{rwy}'"))
        rd.append(
            {"rwy":rwy,
             "nrf":nrf}
        )
        if rwy in run_dict.keys():
            nrf = run_dict[rwy] +nrf
        run_dict[rwy] = nrf
    rd = pd.DataFrame.from_dict(rd)
    table.append({
        "cluster":df_centr.cluster.values[0],
        "num_flights":len(t),
        "centroid_id": df_centr.flight_id.values[0],
        "mean_vertical_rate": np.mean(np.array([f.data.vertical_rate.mean() for f in t])),
        "mean_final_altitude": np.mean(np.array([f.data.altitude.max() for f in t])),
        "mean_groundspeed": np.mean(np.array([f.data.groundspeed.mean() for f in t])),
        "mean_duration": np.mean(np.array([f.data.ts.max() for f in t])),
        "mean_distance": np.mean(np.array([f.data.distance.max() for f in t])),
        "mean_final_latitude": np.mean(np.array([f.data.latitude.values[-1] for f in t])),
        "mean_final_longitude": np.mean(np.array([f.data.longitude.values[-1] for f in t])),
        # "runways": rd,
        "max_rwy":rd.query("nrf==nrf.max()").rwy.values[0],
        "max_rwy_flights":pd.DataFrame.from_dict(rd).nrf.max(),
    })
df_tab = pd.DataFrame.from_dict(table).assign(idd_max_rwy = max(run_dict, key=run_dict.get),
        idd_max_rwy_flights= max(run_dict.values()))
df_tab.to_csv(f"data_generated/clusters_table_{idd}.csv", index=False)
df_tab

# %%
