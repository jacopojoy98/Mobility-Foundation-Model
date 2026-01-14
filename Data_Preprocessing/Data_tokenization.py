import os 
import pandas as pd
import geopandas as gpd

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
            self.raw_voyages_data = gpd.GeoDataFrame
    
    def load_trajectory_data(self) -> gpd.GeoDataFrame:
        _, extension = os.path.splitext(self.trajectory_data_path)
        if extension == '.parquet': 
            self.raw_trajectory_data = gpd.read_parquet(self.trajectory_data_path)
        elif extension == '.csv':
            self.raw_trajectory_data = pd.read_csv(self.trajectory_data_path)
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
    tajectory_path = "./Datasets/nyc_merged.parquet"
    data=Data(trajectory_data_path= tajectory_path)
    data.load_trajectory_data()
    data.osm_data_from_trajectory()