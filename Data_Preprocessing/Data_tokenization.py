import os 
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from Tokenizer import voyages_tokenizer, trajectory_tokenizer
from Tokenizer_functions_old import load_osm_data

class Data():
    def __init__(self, voyages_data_path: str = None, trajectory_data_path: str = None):
        self.voyages_data_path = voyages_data_path
        self.trajectory_data_path = trajectory_data_path
    
    def load_voyages_data(self) -> gpd.GeoDataFrame:     
        _, extension = os.path.splitext(self.voyages_data_path)
        if extension == '.parquet':
            self.raw_voyages_data = gpd.read_parquet(self.voyages_data_path)
        if extension == '.csv':
            pandas_raw_voyages_data = pd.read_csv(self.voyages_data_path)
            self.raw_voyages_data = gpd.GeoDataFrame(
                pandas_raw_voyages_data,            
                geometry = gpd.points_from_xy(pandas_raw_voyages_data["lon"]                                              ,
                                              pandas_raw_voyages_data["lat"]),
               crs = "EPSG:4326"
            )
    
    def load_trajectory_data(self) -> gpd.GeoDataFrame:
        _, extension = os.path.splitext(self.trajectory_data_path)
        if extension == '.parquet': 
            self.raw_trajectory_data = gpd.read_parquet(self.trajectory_data_path)
        elif extension == '.csv':
            pandas_raw_trajectory_data = pd.read_csv(self.trajectory_data_path)
            self.raw_trajectory_data = gpd.GeoDataFrame(
                pandas_raw_trajectory_data,
                geometry=gpd.points_from_xy(pandas_raw_trajectory_data["lon"],
                                            pandas_raw_trajectory_data["lat"]),
                crs = "EPSG:4326"                                            
            )
        # if set(self.raw_trajectory_data.columns) != {'uid', 'lat', 'lon', 'timestamp'}:
            # raise ValueError(f'trajectory data columns are {self.raw_trajectory_data.colums}, expected "uid", "lat", "lon", "timestamp"')

    def osm_data_from_trajectory(self):
        G, gdf_edges, gdf_nodes, intersections, pois_points, landuse = load_osm_data(self.raw_trajectory_data)
        self.G = G
        self.gdf_edges = gdf_edges
        self.gdf_nodes = gdf_nodes
        self.intersections = intersections
        self.pois_points = pois_points
        self.landuse = landuse

    def tokenize_voyages_data(self):
        self.tokenized_voyages_data = voyages_tokenizer(self.raw_voyages_data)

    def tokenize_trajectory_data(self):
        self.tokenized_trajectory_data = trajectory_tokenizer(self.raw_trajectory_data)


## TEST

if __name__ == "__main__":
    # a = np.arange(0,10)
    # b = np.arange(10,20)
    # c = np.arange(20,30)
    # d = np.stack([a,b,c], axis = -1)
    # e_0 = np.concatenate([a,b,c], axis = 0)
    # e_1 = np.concatenate([a,b,c], axis = -1)
    # f = np.hstack([a,b,c])
    # print(f"concatenate_0: {e_0}")
    # print(f"concatenate_1: {e_1}")
    # print(f"stack_-1: {d}")
    # print(f"hstack: {f}")
    # exit()  
    tajectory_path = "~/Modelli/Datasets/nyc_merged_preprocessed_stops.parquet"
    data = Data(trajectory_data_path = tajectory_path)
    data.load_trajectory_data()
    data.osm_data_from_trajectory()
    tokens = trajectory_tokenizer(data.raw_trajectory_data, data.G, data.pois_points, data.landuse)
    with open('preliminary_tokens.pkl','wb') as f:
        pickle.dump(tokens, f)