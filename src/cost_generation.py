#%%
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
#%%
def downsample_array(pop_x, max_length=270):
    if len(pop_x) <= max_length:
        return pop_x
    indices = np.linspace(0, len(pop_x) - 1, max_length, dtype=int)
    reduced_pop_x = pop_x[indices]

    return reduced_pop_x

def downsample_1d(array):
    """
    Downsample a 1D array by averaging pairs of elements.
    For arrays with odd length, the last element is preserved.
    
    Parameters:
    array: numpy 1D array of any length
    
    Returns:
    numpy 1D array of length ceil(len(array)/2)
    """
    length = len(array)
    new_length = (length + 1) // 2  # Ceiling division to handle odd lengths
    
    result = np.zeros(new_length, dtype=array.dtype)
    
    # Handle the evenly paired elements
    even_length = (length // 2) * 2
    if even_length > 0:
        # Reshape to group pairs, then average
        paired = array[:even_length].reshape(-1, 2)
        result[:length//2] = paired.mean(axis=1)
    
    # Handle the last odd element if present
    if length % 2 == 1:
        result[-1] = array[-1]
    
    return result
def downsample_coords(matrix):
    """
    Downsample a numpy matrix of cordinates by finding a center of every 2×2 block of 4 points.
    For matrices with odd dimensions, the last row and/or column will be preserved.
    
    Parameters:
    matrix: numpy array of shape (n, m, k), n-rows, m-columns, k-x and y coordinates
    
    Returns:
    numpy array of shape (n//2 + n%2, m//2 + m%2, k)
    """
        
    rows, cols,xy = matrix.shape
    
    # Calculate new dimensions
    new_rows = rows // 2 + (rows % 2)
    new_cols = cols // 2 + (cols % 2)
    
    # Create result array
    result = np.zeros((new_rows, new_cols,2), dtype=matrix.dtype)
    
    # Process even-sized part of the matrix using vectorized operations
    even_rows = (rows // 2) * 2
    even_cols = (cols // 2) * 2
    
    # Reshape and sum to avoid loops
    if even_rows > 0 and even_cols > 0:
        # Extract the part of the matrix that can be fully processed in 2×2 blocks
        even_part = matrix[:even_rows, :even_cols]
        
        # Reshape to group 2×2 blocks, then sum along the appropriate axes
        reshaped = even_part.reshape(rows//2, 2, cols//2, 2,2)
        result[:rows//2, :cols//2,0] = reshaped[:,:,:,:,0].sum(axis=(1, 3))/4
        result[:rows//2, :cols//2,1] = reshaped[:,:,:,:,1].sum(axis=(1, 3))/4
        # result[:rows//2, :cols//2,:] = np.array([reshaped[:,:,:,:,0].sum(axis=(1, 3))/4,reshaped[:,:,:,:,1].sum(axis=(1, 3))/4]).reshape(result[:rows//2, :cols//2,:].shape)
    
    # Handle the last row if odd number of rows
    if rows % 2 == 1:
        last_row = matrix[rows-1, :even_cols,:]
        result[new_rows-1, :cols//2,:] = last_row.reshape(-1, 2,2).sum(axis=1)/2
    
    # Handle the last column if odd number of columns
    if cols % 2 == 1:
        last_col = matrix[:even_rows, cols-1,:]
        result[:rows//2, new_cols-1,:] = last_col.reshape(-1, 2,2).sum(axis=1)/2
    
    # Handle corner element if both dimensions are odd
    if rows % 2 == 1 and cols % 2 == 1:
        result[new_rows-1, new_cols-1,:] = matrix[rows-1, cols-1,:]
    
    return result
def downsample_map(matrix):
    """
    Downsample a numpy matrix by merging every 2×2 block of 4 points.
    For matrices with odd dimensions, the last row and/or column will be preserved.
    
    Parameters:
    matrix: numpy array of shape (n, m)
    
    Returns:
    numpy array of shape (n//2 + n%2, m//2 + m%2)
    """
    rows, cols = matrix.shape
    
    # Calculate new dimensions
    new_rows = rows // 2 + (rows % 2)
    new_cols = cols // 2 + (cols % 2)
    
    # Create result array
    result = np.zeros((new_rows, new_cols), dtype=matrix.dtype)
    
    # Process even-sized part of the matrix using vectorized operations
    even_rows = (rows // 2) * 2
    even_cols = (cols // 2) * 2
    
    # Reshape and sum to avoid loops
    if even_rows > 0 and even_cols > 0:
        # Extract the part of the matrix that can be fully processed in 2×2 blocks
        even_part = matrix[:even_rows, :even_cols]
        
        # Reshape to group 2×2 blocks, then sum along the appropriate axes
        reshaped = even_part.reshape(rows//2, 2, cols//2, 2)
        result[:rows//2, :cols//2] = reshaped.sum(axis=(1, 3))
    
    # Handle the last row if odd number of rows
    if rows % 2 == 1:
        last_row = matrix[rows-1, :even_cols]
        result[new_rows-1, :cols//2] = last_row.reshape(-1, 2).sum(axis=1)
    
    # Handle the last column if odd number of columns
    if cols % 2 == 1:
        last_col = matrix[:even_rows, cols-1]
        result[:rows//2, new_cols-1] = last_col.reshape(-1, 2).sum(axis=1)
    
    # Handle corner element if both dimensions are odd
    if rows % 2 == 1 and cols % 2 == 1:
        result[new_rows-1, new_cols-1] = matrix[rows-1, cols-1]
    
    return result


def pop_map_preprocessing(pop_map, grid_type, bounds, max_len=230):
    '''
    The map is cropped to the bounds and rescaled to the grid_type.
    pop_map: pd.DataFrame, the population map with columns ["x", "y", "pp"] or ["lon", "lat", "pp"]
    grid_type: "xy" or "ll", "xy" - grid based on x and y coordinates, "ll" - grid based on longitude and latitude
    bounds: list 4 floats, [min_x, min_y, max_x, max_y] or [min_lon, min_lat, max_lon, max_lat]
    max_len: int, the maximum length of the grid, Deault = 230

    returns: pd.DataFrame with columns ["x", "y", "pp"], the rescaled and cropped population map of max size (max_len, max_len),
    if grid_type == "ll", the coordinates are transformed to x and y based on the EPSG:3035 projection
    '''
    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
    transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)
    if np.max(bounds)>180:
        min_x, min_y, max_x, max_y = bounds
        min_lon, min_lat = transformer_ll.transform(bounds[0], bounds[1])
        max_lon, max_lat = transformer_ll.transform(bounds[2], bounds[3])
    else:
        min_lon, min_lat, max_lon, max_lat = bounds
        min_x, min_y = transformer_xy.transform(bounds[0], bounds[1])
        max_x, max_y = transformer_xy.transform(bounds[2], bounds[3])

    if grid_type == "xy":
        pop_map = pop_map.query("@min_x<x<@max_x and @min_y<y<@max_y").fillna(0)
        pop_map.loc[:,"y"] = pop_map["y"].apply(lambda x: round(x, 7))
        pop_map.loc[:,"x"] = pop_map["x"].apply(lambda x: round(x, 7))
        pop_y = pop_map.y.unique()
        pop_x = pop_map.x.unique()
        Lat, Lon = np.meshgrid(pop_y, pop_x)
        new_df = pd.DataFrame(
            np.array([Lat.flatten(), Lon.flatten()]).T, columns=["y", "x"]
        )
        pop_map = new_df.merge(pop_map, on=["y", "x"], how="outer").fillna(0).sort_values(["y","x"]).reset_index(drop=True)
        pop_x = pop_map.x.unique()
        pop_y = pop_map.y.unique()
        max_len = min(max_len, len(pop_x), len(pop_y))
        pop_grid = pop_map.pp.values.reshape(len(pop_y),len(pop_x))
        while max(pop_grid.shape)>max_len:
            x_new = downsample_1d(pop_x)
            y_new = downsample_1d(pop_y)
            pop_coarse = downsample_map(pop_grid)
            pop_grid = pop_coarse
            pop_x = x_new
            pop_y = y_new
        rpp_x, rpp_y = np.meshgrid(x_new, y_new)
        red_pop_map = pd.DataFrame(
        np.vstack([rpp_x.flatten(), rpp_y.flatten(), pop_grid.flatten()]).T,
            columns=["x", "y", "pp"],
        ).sort_values(["y","x"]).reset_index(drop=True)
        Y2d, X2d, Z2d = np.meshgrid(y_new, x_new, [0], indexing="ij")
        point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T
    else:
        pop_map = pop_map.query("@min_lon<lon<@max_lon and @min_lat<lat<@max_lat")
        pop_map.loc[:,"lat"] = pop_map["lat"].apply(lambda x: round(x, 7))
        pop_map.loc[:,"lon"] = pop_map["lon"].apply(lambda x: round(x, 7))
        pop_lat = pop_map.lat.unique()
        pop_lon = pop_map.lon.unique()
        Lat, Lon = np.meshgrid(pop_lat, pop_lon)
        new_df = pd.DataFrame(
            np.array([Lat.flatten(), Lon.flatten()]).T, columns=["lat", "lon"]
        )
        pop_map = new_df.merge(pop_map, on=["lat", "lon"], how="outer").fillna(0).sort_values(["lat","lon"]).reset_index(drop=True)
        pop_lon = pop_map.lon.unique()
        pop_lat = pop_map.lat.unique()
        max_len = min(max_len, len(pop_lon), len(pop_lat))
        x,y = transformer_xy.transform(pop_map.lon, pop_map.lat)
        pop_coords = np.vstack([x,y]).T.reshape(len(pop_lat),len(pop_lon),2)
        pop_grid = pop_map.pp.values.reshape(len(pop_lat),len(pop_lon))
        while max(pop_grid.shape)>max_len:
            pop_coarse = downsample_map(pop_grid)
            pop_grid = pop_coarse
            pop_coords_coarse = downsample_coords(pop_coords)
            pop_coords = pop_coords_coarse
        Lon2d,Lat2d = transformer_ll.transform(pop_coords[:,:,0], pop_coords[:,:,1])
        Z2d = np.zeros(Lon2d.shape)
        red_pop_map = pd.DataFrame(
            np.vstack([Lat2d.flatten(), Lon2d.flatten(), pop_grid.flatten()]).T,
                columns=["lat", "lon", "pp"],
            )#.sort_values(["lat","lon"]).reset_index(drop=True)
        X2d, Y2d = transformer_xy.transform(Lon2d,Lat2d)
        red_pop_map = red_pop_map.assign(x = X2d.flatten(),y = Y2d.flatten())#.sort_values(["y","x"]).reset_index(drop=True)
        point2d = np.array([X2d, Y2d, Z2d]).reshape(3, -1).T
    return red_pop_map, point2d

def cost_generator(
        pop_map, 
        grid_type, 
        airport,
        alt_max = 20000,  
        nodes = (30,20,20), 
        bounds = None,
        max_res_len=230, 
        plot = False):
    '''
    Generation of a cost grid using population map.
    pop_map: pd.DataFrame, the population map with columns ["x", "y", "pp"] and/or ["lon", "lat", "pp"]
    grid_type: "xy" or "ll", "xy" - grid based on x and y coordinates, "ll" - grid based on longitude and latitude
    airport: str, the ICAO code of the airport
    bounds: list of 4 floats, [min_x, min_y, max_x, max_y] or [min_lon, min_lat, max_lon, max_lat]
    nodes: tuple of 3 ints, the number of nodes in x, y and z directions
    alt_max: int, the maximum altitude of the cost grid in meters, default = 20000m
    max_res_len: int, the maximum length of the resample grid, Default = 230
    plot: bool, if True, the plot of the cost grid and poplation map is shown, Default = False
    return: pd.DataFrame with columns ["longitude", "latitude", "altitude", "cost"], the cost grid
    
    '''

    nx, ny, nz = nodes
    airport = nav.airport(airport)

    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)
    transformer_ll = Transformer.from_crs(crs_3035, crs_4326, always_xy=True)

    start = (airport["lat"], airport["lon"])
    start_xy = transformer_xy.transform(start[1], start[0])
    if bounds is not None:
        if np.max(bounds)>180:
            min_x, min_y, max_x, max_y = bounds
            min_lon, min_lat = transformer_ll.transform(bounds[0], bounds[1])
            max_lon, max_lat = transformer_ll.transform(bounds[2], bounds[3])
        else:
            min_lon, min_lat, max_lon, max_lat = bounds
            min_x, min_y = transformer_xy.transform(bounds[0], bounds[1])
            max_x, max_y = transformer_xy.transform(bounds[2], bounds[3])
    else:
        min_x,min_y = start_xy[0] - 300000,start_xy[1] - 300000 # 300 km from the airport
        max_x,max_y = start_xy[0] + 300000,start_xy[1] + 300000 # 300 km from the airport
        min_lon, min_lat = transformer_ll.transform(min_x, min_y)
        max_lon, max_lat = transformer_ll.transform(max_x, max_y)

    min_x, min_y = transformer_xy.transform(min_lon,min_lat)
    max_x, max_y = transformer_xy.transform(max_lon,max_lat)
    red_pop_map, point2d = pop_map_preprocessing(pop_map, grid_type, bounds=[min_x, min_y, max_x, max_y], max_len=max_res_len)

    lonp = np.linspace(min_lon, max_lon, nx)
    latp = np.linspace(min_lat, max_lat, ny)
    yp = np.linspace(min_y, max_y, ny)
    xp = np.linspace(min_x, max_x, nx)
    altp = np.linspace(0, alt_max+2000, nz)
    X, Y, Z = np.meshgrid(xp, yp, altp * aero.ft, indexing="ij") 
    # Lon, Lat = transformer_ll.transform(X, Y)
    point3d = np.array([X, Y, Z]).reshape(3, -1).T

    Lon_cost, Lat_cost, Alt_cost = np.meshgrid(lonp, latp, altp, indexing="ij")
    X_cost, Y_cost = transformer_xy.transform(Lon_cost, Lat_cost)
    Z_cost = Alt_cost * aero.ft


    dist = distance_matrix(point3d, point2d)

    noise_matrix = 1 / dist**2

    combined_cost = np.dot(
        noise_matrix,
        red_pop_map.pp.values,
    )

    cost_grid = combined_cost.reshape(nx, ny, nz)
    interp = RegularGridInterpolator(
        [xp, yp, altp * aero.ft], cost_grid, bounds_error=False, method="linear"
    )  # "linear", "nearest", "slinear", "cubic", "quintic" and "pchip"
    
    cost_interpolated = (
        interp(np.vstack([X_cost.flatten(), Y_cost.flatten(), Z_cost.flatten()]).T)
        .flatten()
        .reshape(nx, ny, nz)
    )
    # cost_grid = cost_interpolated
    df_cost = (
        (
            pd.DataFrame(
                np.array([Lon_cost, Lat_cost, Alt_cost, cost_interpolated])
                .reshape(4, -1)
                .T,
                columns=["longitude", "latitude", "altitude", "cost"],
            )
            .assign(height=lambda x: x.altitude * 0.3048)
            .assign(cost=lambda x: np.where(x.altitude > alt_max, 0, x.cost))
        )
        .sort_values(by=["longitude", "latitude", "height"])
        .fillna(0)
    )

    if plot:
        proj = ccrs.TransverseMercator(
            central_longitude=start[1], central_latitude=start[0]
        )
        # proj = ccrs.PlateCarree()
        trans = ccrs.PlateCarree()

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(6,6),
            subplot_kw=dict(projection=proj),
        )
        ax.set_extent(
            [
                min_lon - 0.1,
                max_lon + 0.1,
                min_lat - 0.1,
                max_lat + 0.1,
            ]
        )
        ax.add_feature(BORDERS, linestyle="solid", alpha=1)
        ax.add_feature(COASTLINE, linestyle="solid", alpha=1)

        norm = plt.Normalize(vmin=0.000104, vmax=0.0003)

        ax.contour(
            transformer_ll.transform(X[:, :, 0], Y[:, :, 0])[0],
            transformer_ll.transform(X[:, :, 0], Y[:, :, 0])[1],
            cost_grid[:, :, 4],
            levels=40,
            alpha=0.5,
            norm=norm,
            transform=trans,
            cmap="cool",
        )

        ax.scatter(
            pop_map.query(f"20000>pp >= 100").lon,
            pop_map.query(f"20000>pp >= 100").lat,
            c=pop_map.query(f"20000>pp >= 100").pp,
            s=1,
            alpha=1,
            transform=trans,
            cmap="Reds",
            norm=plt.Normalize(vmin=100, vmax=10000),
        )
        ax.scatter(

            transformer_ll.transform(    red_pop_map.query(f"20000>pp >= 0").x,
            red_pop_map.query(f"20000>pp >= 0").y,)[0],
            transformer_ll.transform(    red_pop_map.query(f"20000>pp >= 0").x,
            red_pop_map.query(f"20000>pp >= 0").y,)[1],
            c=red_pop_map.query(f"20000>pp >= 0").pp,
            s=1,
            alpha=1,
            transform=trans,
            cmap="viridis",
            norm=plt.Normalize(vmin=100, vmax=10000),
        )
        ax.gridlines(color="k", draw_labels=True, alpha=0.1)

        plt.tight_layout()
        plt.savefig("cost_map.png", bbox_inches = "tight",dpi=300)
        plt.show()
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
