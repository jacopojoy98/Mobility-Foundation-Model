import os 
import pandas as pd

from Tokenizer import voyages_tokenizer, trajectory_tokenizer

class Data():
    def __init__(self, voyages_data_path: str , trajectory_data_path: str):
        self.voyages_data_path = voyages_data_path
        self.trajectory_data_path = trajectory_data_path
    
    def load_voyages_data(self):
        _, extension = os.path.splitext(self.voyages_data_path)
        if extension == '.csv':
            self.raw_voyages_data = pd.read_csv(self.voyages_data_path)
    
    def load_trajectory_data(self):
        _, extension = os.path.splitext(self.trajectory_data_path)
        if extension == '.csv':
            self.raw_trajectory_data = pd.read_csv(self.trajectory_data_path)
        if set(self.raw_trajectory_data.colums) != {'uid', 'lat', 'lon', 'timestamp'}:
            raise ValueError(f'trajectory data columns are {self.raw_trajectory_data.colums}, expected "uid", "lat", "lon", "timestamp"')

    def tokenize_voyages_data(self):
        self.tokenized_voyages_data = voyages_tokenizer(self.raw_voyages_data)

    def tokenize_trajectory_data(self):
        self.tokenized_trajectory_data = trajectory_tokenizer(self.raw_trajectory_data)
    