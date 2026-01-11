import os 
import pandas as pd

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
        
    def tokenize_voyages_data(self):
