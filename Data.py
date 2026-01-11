'''
Definition of the data class:
    Main Routines (not in order):
        - trajectory_features_abstraction
        - voyages_features_abstraction
        - raw_voyages_data_load
        - raw_trajectory_data_load

    Main Attributes:
        - abstracted_trajectory_features
        - abstracted_voyages_features
'''
import os
import pandas as pd 

class Data:
    def __init__(self, path_to_raw_trajectory_data = None, path_to_raw_voyages_data = None):
        self.path_to_raw_voyages_data = path_to_raw_voyages_data
        self.path_to_raw_trajectory_data = path_to_raw_trajectory_data

    def load_raw_voyages_data(self, path_to_raw_voyages_data):
        filename, file_extension = os.path.splitext(path_to_raw_voyages_data)
        if file_extension == '.csv':
            self.raw_voyages_data = pd.read_csv(path_to_raw_voyages_data)
        else:
            raise TypeError("File not in .csv extension")
            
    def load_raw_trajecctory_data(self, path_to_raw_trajectory_data):
        filename, file_extension = os.path.splitext(path_to_raw_trajectory_data)
        if file_extension == '.csv':
            self.raw_trajectory_data = pd.read_csv(path_to_raw_trajectory_data)
        else:
            raise TypeError("File not in .csv extension")

    def tokenize_voyage_data(self, voyage_data_tokenization_attributes):
        self.tokenzed_voyage_data = voyage_data_tokenizer(self.raw_voyages_data)
