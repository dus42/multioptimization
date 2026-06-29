# %%
from traffic.core import Traffic, Flight
from traffic.data import airports
from openap import Thrust, FuelFlow, aero, prop
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from cartes.crs import LCCEurope as lcc
from cartes.crs import EuroPP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


# %%
map_type = "DN"
if map_type == "DN":
    t = Traffic(
        # pd.read_parquet(
        #     "data_generated/opensky_flights_2024dn_ready_for_clustering.parquet"
        # )
        pd.read_parquet("data_generated/for_cluster.parquet")
    )
else:
    t = Traffic(
        pd.read_parquet(
            "data_generated/opensky_flights_2024dn_ready_for_clustering.parquet"
        )
    ).query(f"day_night=='{map_type}'")
# ids = []
# for f in t:
#     if len(f) != 100:
#         ids.append(f.flight_id)
t = (
    # t.query(f"flight_id not in {ids}")
    t.assign(h=lambda x: x.altitude * aero.ft)
    # .assign(month=lambda x: x.timestamp.values.astype("datetime64[M]"))
)
# t.data["month"] = t.data.month.astype(str)
# t.data["month"] = t.data["month"].apply(lambda x: x.split("-")[1])
# pd.DataFrame(list(t.data.callsign.unique()), columns=["callsign"]).to_csv(
#     "callsigns.csv", index=False
# )
# st = np.stack(list(f.data[["latitude", "longitude", "fl"]].values.ravel() for f in t))
# %%
t = t.query("distance<220").query("51.5<latitude<53.2 and 3.8<longitude<5.7")

# %%
# t_dbscan = t.clustering(
#     nb_samples=20,
#     projection=lcc(),
#     features=['latitude', 'longitude', "h"],
#     clustering=DBSCAN(eps=2.2, min_samples=5),
#     transform=StandardScaler(),
# ).fit_predict()

t_dbscan = t.clustering(
    nb_samples=150,
    projection=EuroPP(),
    clustering=DBSCAN(eps=1.5, min_samples=10),
    transform=StandardScaler(),
).fit_predict()
n_clusters = t_dbscan.data.cluster.max() + 1
print("Estimated number of clusters: %d" % n_clusters)
print("Estimated number of noise points: %d" % len(t_dbscan.query("cluster==-1")))
# print(
#     "Estimated number of noise fligths: %d"
#     % (len(t_dbscan.data.query("cluster==-1")) / 100)
# )
# t_dbscan = t_dbscan.query("cluster!=-1")
t_dbscan.data.cluster.hist()
# %%
saved_indices = []
for cluster in range(n_clusters):

    current_cluster = t_dbscan.query(f"cluster == {cluster}")
    ids = list(f.flight_id for f in current_cluster)
    indexx = current_cluster.centroid(
        40,
        projection=EuroPP(),
    ).flight_id  # Not sure about this centroid search, "copied" and adapted from traffic's git repo
    saved_indices.append(indexx)
# # %%

# colors = list(mcolors.TABLEAU_COLORS.keys())
# colors = np.array(list(mcolors.CSS4_COLORS.keys()))
# colors = colors[range(1, 148, 3)]
# # colors.extend(["r", "b", "g", "y", "m", "c"])


# for inx in saved_indices:
#     fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(6, 6))
#     df = t_dbscan[inx].data
#     for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
#         ax0.plot(
#             f.data.longitude,
#             f.data.altitude,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.set_title(
#         f"cluster {df.cluster.values[0]}, {len(t_dbscan.query(f"cluster=={df.cluster.values[0]}"))}"
#     )
#     for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
#         ax1.plot(
#             f.data.ts,
#             f.data.vertical_rate,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.set_title("vertical_rate")
#     for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
#         ax2.plot(
#             f.data.ts,
#             f.data.groundspeed,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.set_title("groundspeed")
#     # for f in t_dbscan.query(f"cluster=={df.cluster.values[0]}"):
#     #     ax3.plot(
#     #         f.data.ts,
#     #         f.data.thrust,
#     #         color=colors[f.data.cluster.values[0]],
#     #         lw=2,
#     #         linestyle="dashed",
#     #         alpha=0.3,
#     #     )
#     # ax3.plot(
#     #     df.ts,
#     #     df.thrust,
#     #     color="w",
#     #     lw=3.5,
#     #     # linestyle="dashed",
#     #     alpha=1,
#     # )
#     # ax3.plot(
#     #     df.ts,
#     #     df.thrust,
#     #     color="k",
#     #     lw=3,
#     #     # linestyle="dashed",
#     #     alpha=1,
#     # )
#     # ax3.plot(
#     #     df.ts,
#     #     df.thrust,
#     #     color=colors[df.cluster.values[0]],
#     #     lw=2,
#     #     # linestyle="dashed",
#     #     alpha=1,
#     # )
#     ax3.set_title("thrust")
#     plt.show()

# %%
if map_type == "DN":
    t_dbscan = t_dbscan.query("cluster not in [-1]")  ### DN
elif map_type == "D":
    t_dbscan = t_dbscan.query("cluster not in [-1,15,19]")  ### D
else:
    t_dbscan = t_dbscan.query("cluster not in [-1,5]")  ### N
t = Traffic(t_dbscan.data.drop(columns=["cluster"]))
# ########################################################################################
# # %%
# t_dbscan_1 = t.clustering(
#     nb_samples=100,
#     # projection=lcc(),
#     features=["x", "y","h"],
#     clustering=DBSCAN(eps=4.3, min_samples=3),
#     transform=StandardScaler(),
# ).fit_predict()

t_dbscan_1 = t.clustering(
    nb_samples=150,
    projection=EuroPP(),
    clustering=DBSCAN(eps=8.5, min_samples=10),
    transform=StandardScaler(),
).fit_predict()
n_clusters = t_dbscan_1.data.cluster.max() + 1
# print("Estimated number of clusters: %d" % n_clusters)
# print(
#     "Estimated number of noise points: %d" % len(t_dbscan_1.query("cluster==-1"))
# )
# print(
#     "Estimated number of noise fligths: %d"
#     % (len(t_dbscan_1.data.query("cluster==-1")) / 100)
# )
t_dbscan_1 = t_dbscan_1.query("cluster!=-1")
t_dbscan_1.data.cluster.hist()
# %%
saved_indices = []
for cluster in range(n_clusters):

    current_cluster = t_dbscan_1.query(f"cluster == {cluster}")
    ids = list(f.flight_id for f in current_cluster)
    indexx = current_cluster.centroid(
        40,
        projection=lcc(),
    ).flight_id  # Not sure about this centroid search, "copied" and adapted from traffic's git repo
    saved_indices.append(indexx)
# # %%

# colors = list(mcolors.TABLEAU_COLORS.keys())
# colors = np.array(list(mcolors.CSS4_COLORS.keys()))
# colors = colors[range(1, 148, 5)]
# # colors.extend(["r", "b", "g", "y", "m", "c"])


# for inx in saved_indices:
#     fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(6, 6))
#     df = t_dbscan_1[inx].data
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax0.plot(
#             f.data.longitude,
#             f.data.altitude,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.set_title(
#         f"cluster {df.cluster.values[0]}, {len(t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"))}"
#     )
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax1.plot(
#             f.data.ts,
#             f.data.vertical_rate,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.set_title("vertical_rate")
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax2.plot(
#             f.data.ts,
#             f.data.groundspeed,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.set_title("groundspeed")
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax3.plot(
#             f.data.ts,
#             f.data.thrust,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax3.plot(
#         df.ts,
#         df.thrust,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax3.plot(
#         df.ts,
#         df.thrust,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax3.plot(
#         df.ts,
#         df.thrust,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax3.set_title("thrust")
# plt.show()


# %%
ends = []
t_cnts = t_dbscan_1.query(f"flight_id in {saved_indices}").query("cluster!=-1")
for f in t_cnts:
    ends.append(
        [
            f.data.cluster.values[0],
            f.data.flight_id.values[0],
            f.data.latitude.values[-1],
            f.data.longitude.values[-1],
            f.data.tow.values[-1],
            f.data.altitude.values[-1],
            f.data.runway.values[-1],
            # f.data.month.values[-1],
        ]
    )
ends = pd.DataFrame(
    ends,
    columns=[
        "cluster",
        "fid",
        "latitude",
        "longitude",
        "tow",
        "altitude",
        "runway",
        # "month",
    ],
)

ends.to_csv(f"data_generated/opensky_centroid_ends_{map_type}_part.csv", index=False)
t_cnts.to_parquet(
    f"data_generated/opensky2024_centroids_{map_type}_part.parquet", index=False
)
t_dbscan_1.to_parquet(
    f"data_generated/opensky2024_clustered_flights_{map_type}_part.parquet", index=False
)
# # %%
# colors = list(mcolors.TABLEAU_COLORS.keys())
# colors.extend(["r", "b", "g", "y", "m", "c"])
# ax = plt.subplot()

# for f in t_dbscan_1:
#     ax.plot(
#         f.data.longitude,
#         f.data.latitude,
#         color=colors[f.data.cluster.values[0]],
#         lw=2,
#         linestyle="dashed",
#         alpha=0.1,
#     )
# for inx in saved_indices:
#     df = t_dbscan_1[str(inx)].data
#     ax.plot(
#         df.longitude,
#         df.latitude,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )

# # %%
# map_type = "DN"
# t_dbscan_1 = Traffic(
#     pd.read_parquet(f"data_generated/opensky2024_clustered_flights_{map_type}.parquet")
# )
# ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
# table = []
# run_dict = {"idd": 1}
# for inx in ends.fid:
#     df_centr = t_dbscan_1[str(inx)].data
#     t = t_dbscan_1.query(f"cluster=={df_centr.cluster.values[0]}")
#     runways = t.data.runway.unique()
#     rd = []
#     for rwy in runways:
#         nrf = len(t.query(f"runway == '{rwy}'"))
#         rd.append({"rwy": rwy, "nrf": nrf})
#         if rwy in run_dict.keys():
#             nrf = run_dict[rwy] + nrf
#         run_dict[rwy] = nrf
#     rd = pd.DataFrame.from_dict(rd)
#     table.append(
#         {
#             "cluster": df_centr.cluster.values[0],
#             "num_flights": len(t),
#             "centroid_id": df_centr.flight_id.values[0],
#             "mean_vertical_rate": np.mean(
#                 np.array([f.data.vertical_rate.mean() for f in t])
#             ),
#             "mean_final_altitude": np.mean(
#                 np.array([f.data.altitude.max() for f in t])
#             ),
#             "mean_groundspeed": np.mean(
#                 np.array([f.data.groundspeed.mean() for f in t])
#             ),
#             "mean_duration": np.mean(np.array([f.data.ts.max() for f in t])),
#             "mean_distance": np.mean(np.array([f.data.distance.max() for f in t])),
#             "mean_final_latitude": np.mean(
#                 np.array([f.data.latitude.values[-1] for f in t])
#             ),
#             "mean_final_longitude": np.mean(
#                 np.array([f.data.longitude.values[-1] for f in t])
#             ),
#             # "runways": rd,
#             "max_rwy": rd.query("nrf==nrf.max()").rwy.values[0],
#             "max_rwy_flights": pd.DataFrame.from_dict(rd).nrf.max(),
#         }
#     )
# df_tab = pd.DataFrame.from_dict(table).assign(
#     idd_max_rwy=max(run_dict, key=run_dict.get),
#     idd_max_rwy_flights=max(run_dict.values()),
# )
# df_tab.to_csv(f"data_generated/clusters_table_{map_type}.csv", index=False)
# df_tab

# # # %%
# # for map_type in ["DN","D","N"]:
# # # for map_type in ["N"]:
# #     if map_type == "DN":
# #         cluster = 4
# #     else:
# #         cluster = 3
# #     t = Traffic(
# #         pd.read_parquet(
# #             f"data_generated/opensky2024_clustered_flights_{map_type}.parquet"
# #             ).query("cluster==@cluster and runway =='24'")
# #     )
# #     t = t[:50]
# #     t.data.to_parquet(f"data_generated/opensky2024_runway24_{map_type}.parquet", index=False)
# #     ends = []
# #     for f in t:
# #         ends.append(
# #             [
# #                 f.data.flight_id.values[0],
# #                 f.data.latitude.values[-1],
# #                 f.data.longitude.values[-1],
# #                 f.data.tow.values[-1],
# #                 f.data.altitude.values[-1],
# #                 f.data.runway.values[-1],
# #             ]
# #         )
# #     ends = pd.DataFrame(
# #         ends, columns=["fid", "latitude", "longitude", "tow", "altitude", "runway"]
# #     )

# #     ends.to_csv(f"data_generated/opensky_runway24_ends_{map_type}.csv", index=False)

# # # %%
# # %%
# rwy = "18L"
# for map_type in ["DN", "D", "N"]:
#     cluster = 0
#     t = Traffic(
#         pd.read_parquet(
#             f"data_generated/opensky2024_clustered_flights_{map_type}.parquet"
#         ).query("cluster==@cluster and runway ==@rwy")
#     )
#     t = t[:50]
#     print(len(t))
#     t.data.to_parquet(
#         f"data_generated/opensky2024_runway18L_{map_type}.parquet", index=False
#     )
#     ends = []
#     for f in t:
#         ends.append(
#             [
#                 f.data.flight_id.values[0],
#                 f.data.latitude.values[-1],
#                 f.data.longitude.values[-1],
#                 f.data.tow.values[-1],
#                 f.data.altitude.values[-1],
#                 f.data.runway.values[-1],
#                 f.data.month.values[-1],
#             ]
#         )
#     ends = pd.DataFrame(
#         ends,
#         columns=["fid", "latitude", "longitude", "tow", "altitude", "runway", "month"],
#     )

#     ends.to_csv(f"data_generated/opensky_runway{rwy}_ends_{map_type}.csv", index=False)

# %%


map_type = "DN"
if map_type == "DN":
    t = Traffic(
        pd.read_parquet("data_generated/for_cluster.parquet").assign(
            h=lambda x: x.altitude * aero.ft
        )
    )
icaos = [
    "3c49c8",
    "3c56e7",
    "3c56e9",
    "3c49ce",
    "3c56ec",
    "461f6d",
    "461f65",
    "461f64",
    "461f6a",
    "461f63",
    "461f67",
    "45ac30",
    "45ac37",
    "45ac2e",
    "45ac32",
    "45ac39",
    "45ac34",
    "45ac33",
    "45ac35",
]

t = t.query(f"icao24 in {icaos}")
## %%
# t=t.resample(100).eval(6, desc="doing")
# t=t.drop_duplicates(subset=("latitude", "longitude", "velocity"))
# %%
# t_dbscan = t.clustering(
#     nb_samples=20,
#     projection=lcc(),
#     features=['latitude', 'longitude', "h"],
#     clustering=DBSCAN(eps=7, min_samples=2),
#     transform=StandardScaler(),
# ).fit_predict()
t_dbscan = t.clustering(
    nb_samples=150,
    projection=EuroPP(),
    clustering=DBSCAN(eps=9, min_samples=3),
    transform=StandardScaler(),
).fit_predict()
n_clusters = t_dbscan.data.cluster.max() + 1
# print("Estimated number of clusters: %d" % n_clusters)
# print("Estimated number of noise points: %d" % len(t_dbscan.query("cluster==-1")))
# print(
#     "Estimated number of noise fligths: %d"
#     % (len(t_dbscan.data.query("cluster==-1")) / 100)
# )
t_dbscan = t_dbscan.query("cluster!=-1")
t_dbscan.data.cluster.hist()

# %%
t_dbscan_1 = t_dbscan
saved_indices = []
for cluster in range(n_clusters):

    current_cluster = t_dbscan_1.query(f"cluster == {cluster}")
    ids = list(f.flight_id for f in current_cluster)
    indexx = current_cluster.centroid(
        40,
        projection=lcc(),
    ).flight_id  # Not sure about this centroid search, "copied" and adapted from traffic's git repo
    saved_indices.append(indexx)
# # %%

# colors = list(mcolors.TABLEAU_COLORS.keys())
# colors = np.array(list(mcolors.CSS4_COLORS.keys()))
# colors = colors[range(1, 148, 5)]
# # colors.extend(["r", "b", "g", "y", "m", "c"])


# for inx in saved_indices:
#     fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(6, 6))
#     df = t_dbscan_1[inx].data
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax0.plot(
#             f.data.longitude,
#             f.data.altitude,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.plot(
#         df.longitude,
#         df.altitude,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax0.set_title(
#         f"cluster {df.cluster.values[0]}, {len(t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"))}"
#     )
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax1.plot(
#             f.data.ts,
#             f.data.vertical_rate,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.plot(
#         df.ts,
#         df.vertical_rate,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax1.set_title("vertical_rate")
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax2.plot(
#             f.data.ts,
#             f.data.groundspeed,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.plot(
#         df.ts,
#         df.groundspeed,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax2.set_title("groundspeed")
#     for f in t_dbscan_1.query(f"cluster=={df.cluster.values[0]}"):
#         ax3.plot(
#             f.data.ts,
#             f.data.thrust,
#             color=colors[f.data.cluster.values[0]],
#             lw=2,
#             linestyle="dashed",
#             alpha=0.3,
#         )
#     ax3.plot(
#         df.ts,
#         df.thrust,
#         color="w",
#         lw=3.5,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax3.plot(
#         df.ts,
#         df.thrust,
#         color="k",
#         lw=3,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax3.plot(
#         df.ts,
#         df.thrust,
#         color=colors[df.cluster.values[0]],
#         lw=2,
#         # linestyle="dashed",
#         alpha=1,
#     )
#     ax3.set_title("thrust")
#     plt.show()


# %%

t_dbscan_1_orig = pd.read_parquet(
    f"data_generated/opensky2024_clustered_flights_{map_type}_part.parquet"
)
maxcl = t_dbscan_1_orig.cluster.max() + 1

t_dbscan_1 = t_dbscan_1.assign(cluster=lambda x: x.cluster + maxcl)
t_dbscan_1 = pd.concat([t_dbscan_1_orig, t_dbscan_1.data])
t_dbscan_1.to_parquet(
    f"data_generated/opensky2024_clustered_flights_{map_type}.parquet", index=False
)
# %%
df = pd.read_parquet(
    f"data_generated/opensky_flights_2024dn_for_centroids.parquet"
).assign(h=lambda x: x.altitude * aero.ft)
# ends = []
ends_orig = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")

t_cnts_orig = pd.read_parquet(
    f"data_generated/opensky2024_centroids_{map_type}_part.parquet"
)
t_cnts = t_dbscan_1.query(f"flight_id in {saved_indices}").query("cluster!=-1")
t_cnts = pd.concat([t_cnts_orig, t_cnts])
df = df.query(f"flight_id in {t_cnts.flight_id.unique().tolist()}")
t_cnts = (
    Traffic(df.merge(t_cnts[["flight_id", "cluster"]], on="flight_id"))
    .drop_duplicates()
    .resample(300)
    .eval(6, desc="resampling")
)
ends = []
for f in t_cnts:
    ends.append(
        [
            f.data.cluster.values[0],
            f.data.flight_id.values[0],
            f.data.latitude.values[-1],
            f.data.longitude.values[-1],
            f.data.tow.values[-1],
            f.data.altitude.max(),
            f.data.altitude.min(),
            f.data.runway.values[-1],
            # f.data.month.values[-1],
        ]
    )
ends = pd.DataFrame(
    ends,
    columns=[
        "cluster",
        "fid",
        "latitude",
        "longitude",
        "tow",
        "altitude",
        "alt_start",
        "runway",
        # "month",
    ],
)

ends.to_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv", index=False)

t_cnts.to_parquet(
    f"data_generated/opensky2024_centroids_{map_type}.parquet", index=False
)

# %%
