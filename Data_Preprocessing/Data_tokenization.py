import os 
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Tuple, Optional, List
import geohash2 as gh  # pip install geohash2
from collections import Counter
from sklearn.cluster import DBSCAN

import pickle
from Tokenizer import voyages_tokenizer, trajectory_tokenizer
from Tokenizer_functions_old import load_osm_data

class Data():
    def __init__(self, voyages_data_path: str = None, trajectory_data_path: str = None):
        self.voyages_data_path = voyages_data_path
        self.trajectory_data_path = trajectory_data_path
        self.evaluation_labels = {}
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

    def extract_destination_labels_geohash(
        self,
        data_type: str = 'trajectory',
        geohash_precision: int = 7,
        uid_column: str = 'uid'
    ) -> Dict:
        """
        Extract destination labels using geohashing.
        
        Geohash precision levels:
        - 5: ~4.9km x 4.9km
        - 6: ~1.2km x 0.6km
        - 7: ~153m x 153m (recommended for destinations)
        - 8: ~38m x 19m
        - 9: ~4.8m x 4.8m
        
        Args:
            data_type: 'trajectory' or 'voyages'
            geohash_precision: Geohash precision level (5-9)
            uid_column: Name of user/trajectory ID column
            
        Returns:
            Dictionary with destination labels and metadata
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING DESTINATION LABELS (Geohash precision={geohash_precision})")
        print(f"{'='*70}")
        
        # Select data
        if data_type == 'trajectory':
            if not hasattr(self, 'raw_trajectory_data'):
                raise ValueError("Load trajectory data first using load_trajectory_data()")
            data = self.raw_trajectory_data
        else:  # voyages
            if not hasattr(self, 'raw_voyages_data'):
                raise ValueError("Load voyages data first using load_voyages_data()")
            data = self.raw_voyages_data
        
        # Group by UID to get trajectories
        trajectories_grouped = data.groupby(uid_column)
        
        destinations = []
        destination_geohashes = []
        trajectory_ids = []
        
        print(f"Processing {len(trajectories_grouped)} trajectories...")
        
        for uid, u_group in trajectories_grouped:
            for traj_id, group in u_group.groupby('traj_id'):
                # Sort by timestamp to get final point
                if 'timestamp' in group.columns:
                    group = group.sort_values('timestamp')
                
                # Get last point
                last_point = group.iloc[-1]
                lat = last_point.geometry.y
                lon = last_point.geometry.x
                
                # Compute geohash
                dest_geohash = gh.encode(lat, lon, precision=geohash_precision)
                
                destinations.append((lat, lon))
                destination_geohashes.append(dest_geohash)
                trajectory_ids.append(traj_id)
        
        # Map geohashes to integer labels
        unique_geohashes = sorted(set(destination_geohashes))
        geohash_to_label = {geohash: idx for idx, geohash in enumerate(unique_geohashes)}
        
        destination_labels = np.array([
            geohash_to_label[gh] for gh in destination_geohashes
        ])
        
        print(f"\nResults:")
        print(f"  Total trajectories: {len(trajectory_ids)}")
        print(f"  Unique destinations: {len(unique_geohashes)}")
        print(f"  Geohash precision: {geohash_precision}")
        
        # Store results
        labels_dict = {
            'task': 'destination_prediction',
            'destination_labels': destination_labels,
            'destination_geohashes': destination_geohashes,
            'destination_coords': destinations,
            'trajectory_ids': trajectory_ids,
            'num_destinations': len(unique_geohashes),
            'geohash_to_label': geohash_to_label,
            'label_to_geohash': {v: k for k, v in geohash_to_label.items()},
            'geohash_precision': geohash_precision
        }
        
        self.evaluation_labels['destination'] = labels_dict
        
        return labels_dict
    
    def extract_time_of_arrival_labels(
        self,
        data_type: str = 'trajectory',
        uid_column: str = 'user',
        timestamp_column: str = 'time',
        normalize: bool = True
    ) -> Dict:
        """
        Extract time of arrival labels.
        
        Args:
            data_type: 'trajectory' or 'voyages'
            uid_column: Name of user/trajectory ID column
            timestamp_column: Name of timestamp column
            normalize: Whether to normalize times to [0, 1]
            
        Returns:
            Dictionary with arrival time labels
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING TIME OF ARRIVAL LABELS")
        print(f"{'='*70}")
        
        # Select data
        if data_type == 'trajectory':
            if not hasattr(self, 'raw_trajectory_data'):
                raise ValueError("Load trajectory data first")
            data = self.raw_trajectory_data
        else:
            if not hasattr(self, 'raw_voyages_data'):
                raise ValueError("Load voyages data first")
            data = self.raw_voyages_data
        
        trajectories_grouped = data.groupby(uid_column)
        
        arrival_times = []
        trajectory_ids = []
        
        print(f"Processing {len(trajectories_grouped)} trajectories...")
        
        for uid, u_group in trajectories_grouped:
            for traj_id, group in u_group.groupby('traj_id'):
                
                # Get last timestamp
                last_timestamp = group.iloc[-1][timestamp_column]
                
                # Convert to numeric if needed
                if isinstance(last_timestamp, str):
                    last_timestamp = pd.to_datetime(last_timestamp)
                
                if isinstance(last_timestamp, pd.Timestamp):
                    # Convert to hour of day (0-24)
                    arrival_time = last_timestamp.hour + last_timestamp.minute / 60.0
                else:
                    arrival_time = float(last_timestamp)
                
                arrival_times.append(arrival_time)
                trajectory_ids.append(traj_id)
        
        arrival_times = np.array(arrival_times)
        
        # Normalize if requested
        if normalize:
            min_time = arrival_times.min()
            max_time = arrival_times.max()
            arrival_times_normalized = (arrival_times - min_time) / (max_time - min_time)
        else:
            arrival_times_normalized = arrival_times
        
        print(f"\nResults:")
        print(f"  Total trajectories: {len(trajectory_ids)}")
        print(f"  Time range: [{arrival_times.min():.2f}, {arrival_times.max():.2f}]")
        if normalize:
            print(f"  Normalized to: [0, 1]")
        
        labels_dict = {
            'task': 'time_of_arrival',
            'arrival_times': arrival_times_normalized,
            'arrival_times_raw': arrival_times,
            'trajectory_ids': trajectory_ids,
            'normalized': normalize
        }
        
        self.evaluation_labels['time_of_arrival'] = labels_dict
        
        return labels_dict
    
    def extract_next_location_labels_geohash(
        self,
        data_type: str = 'trajectory',
        geohash_precision: int = 7,
        uid_column: str = 'uid',
        prefix_ratio: float = 0.8
    ) -> Dict:
        """
        Extract next location labels using geohashing.
        
        Args:
            data_type: 'trajectory' or 'voyages'
            geohash_precision: Geohash precision level
            uid_column: Name of user/trajectory ID column
            prefix_ratio: Ratio of trajectory to use as prefix
            
        Returns:
            Dictionary with next location labels and prefix info
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING NEXT LOCATION LABELS (Geohash precision={geohash_precision})")
        print(f"{'='*70}")
        
        # Select data
        if data_type == 'trajectory':
            if not hasattr(self, 'raw_trajectory_data'):
                raise ValueError("Load trajectory data first")
            data = self.raw_trajectory_data
        else:
            if not hasattr(self, 'raw_voyages_data'):
                raise ValueError("Load voyages data first")
            data = self.raw_voyages_data
        
        trajectories_grouped = data.groupby(uid_column)
        
        next_locations = []
        next_geohashes = []
        prefix_lengths = []
        trajectory_ids = []
        
        print(f"Processing {len(trajectories_grouped)} trajectories...")
        
        for uid, u_group in trajectories_grouped:
            for traj_id, group in u_group.groupby('traj_id'):
                # Calculate prefix length
                traj_length = len(group)
                prefix_len = int(traj_length * prefix_ratio)
                prefix_len = max(1, min(prefix_len, traj_length - 1))
                
                # Get next location after prefix
                next_point = group.iloc[prefix_len]
                lat = next_point.geometry.y
                lon = next_point.geometry.x
                
                # Compute geohash
                next_geohash = gh.encode(lat, lon, precision=geohash_precision)
                
                next_locations.append((lat, lon))
                next_geohashes.append(next_geohash)
                prefix_lengths.append(prefix_len)
                trajectory_ids.append(traj_id)
            
        # Map geohashes to integer labels
        unique_geohashes = sorted(set(next_geohashes))
        geohash_to_label = {geohash: idx for idx, geohash in enumerate(unique_geohashes)}
        
        next_location_labels = np.array([
            geohash_to_label[gh] for gh in next_geohashes
        ])
        
        print(f"\nResults:")
        print(f"  Total trajectories: {len(trajectory_ids)}")
        print(f"  Unique next locations: {len(unique_geohashes)}")
        print(f"  Prefix ratio: {prefix_ratio}")
        
        labels_dict = {
            'task': 'next_location',
            'next_location_labels': next_location_labels,
            'next_geohashes': next_geohashes,
            'next_coords': next_locations,
            'prefix_lengths': prefix_lengths,
            'trajectory_ids': trajectory_ids,
            'num_next_locations': len(unique_geohashes),
            'geohash_to_label': geohash_to_label,
            'label_to_geohash': {v: k for k, v in geohash_to_label.items()},
            'geohash_precision': geohash_precision,
            'prefix_ratio': prefix_ratio
        }
        
        self.evaluation_labels['next_location'] = labels_dict
        
        return labels_dict
    
    def extract_trip_purpose_labels(
        self,
        data_type: str = 'trajectory',
        uid_column: str = 'uid',
        timestamp_column: str = 'timestamp'
    ) -> Dict:
        """
        Extract trip purpose labels based on temporal heuristics.
        
        Purpose categories:
        - 0: Work (weekday morning 7-9am or evening 5-7pm)
        - 1: Home (late evening/night 10pm-6am)
        - 2: Leisure (weekend or weekday midday)
        
        Args:
            data_type: 'trajectory' or 'voyages'
            uid_column: Name of user/trajectory ID column
            timestamp_column: Name of timestamp column
            
        Returns:
            Dictionary with trip purpose labels
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING TRIP PURPOSE LABELS")
        print(f"{'='*70}")
        
        # Select data
        if data_type == 'trajectory':
            if not hasattr(self, 'raw_trajectory_data'):
                raise ValueError("Load trajectory data first")
            data = self.raw_trajectory_data
        else:
            if not hasattr(self, 'raw_voyages_data'):
                raise ValueError("Load voyages data first")
            data = self.raw_voyages_data
        
        trajectories_grouped = data.groupby(uid_column)
        
        purposes = []
        trajectory_ids = []
        
        print(f"Processing {len(trajectories_grouped)} trajectories...")
        
        for uid, u_group in trajectories_grouped:
            for traj_id, group in u_group.groupby('traj_id'):

                # Get arrival timestamp
                last_timestamp = group.iloc[-1][timestamp_column]
                
                # Convert to datetime if needed
                if isinstance(last_timestamp, str):
                    last_timestamp = pd.to_datetime(last_timestamp)
                
                if isinstance(last_timestamp, pd.Timestamp):
                    hour = last_timestamp.hour
                    day_of_week = last_timestamp.dayofweek  # Monday=0, Sunday=6
                    is_weekend = day_of_week >= 5
                else:
                    # Assume timestamp is hour of day
                    hour = int(last_timestamp % 24)
                    is_weekend = False
                
                # Infer purpose
                if not is_weekend:
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        purpose = 0  # Work
                    elif 22 <= hour or hour <= 6:
                        purpose = 1  # Home
                    else:
                        purpose = 2  # Leisure
                else:
                    if 22 <= hour or hour <= 6:
                        purpose = 1  # Home
                    else:
                        purpose = 2  # Leisure
                
                purposes.append(purpose)
                trajectory_ids.append(traj_id)
        
        purposes = np.array(purposes)
        
        # Count distribution
        purpose_counts = Counter(purposes)
        purpose_names = {0: 'Work', 1: 'Home', 2: 'Leisure'}
        
        print(f"\nResults:")
        print(f"  Total trajectories: {len(trajectory_ids)}")
        print(f"  Purpose distribution:")
        for purpose_id, count in sorted(purpose_counts.items()):
            pct = count / len(purposes) * 100
            print(f"    {purpose_names[purpose_id]}: {count} ({pct:.1f}%)")
        
        labels_dict = {
            'task': 'trip_purpose',
            'purpose_labels': purposes,
            'trajectory_ids': trajectory_ids,
            'num_purposes': len(set(purposes)),
            'purpose_names': purpose_names
        }
        
        self.evaluation_labels['trip_purpose'] = labels_dict
        
        return labels_dict
    
    def extract_user_identification_labels(
        self,
        data_type: str = 'trajectory',
        uid_column: str = 'uid'
    ) -> Dict:
        """
        Extract user identification labels.
        
        Args:
            data_type: 'trajectory' or 'voyages'
            uid_column: Name of user/trajectory ID column
            
        Returns:
            Dictionary with user ID labels
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING USER IDENTIFICATION LABELS")
        print(f"{'='*70}")
        
        # Select data
        if data_type == 'trajectory':
            if not hasattr(self, 'raw_trajectory_data'):
                raise ValueError("Load trajectory data first")
            data = self.raw_trajectory_data
        else:
            if not hasattr(self, 'raw_voyages_data'):
                raise ValueError("Load voyages data first")
            data = self.raw_voyages_data
        
        trajectories_grouped = data.groupby(uid_column)
        
        user_ids = []
        trajectory_ids = []
        
        for uid, u_group in trajectories_grouped:
            for traj_id, group in u_group.groupby('traj_id'):
                user_ids.append(uid)
                trajectory_ids.append(traj_id)
        
        # Map user IDs to integer labels
        unique_users = sorted(set(user_ids))
        user_to_label = {user_id: idx for idx, user_id in enumerate(unique_users)}
        
        user_labels = np.array([user_to_label[uid] for uid in user_ids])
        
        print(f"\nResults:")
        print(f"  Total trajectories: {len(trajectory_ids)}")
        print(f"  Unique users: {len(unique_users)}")
        
        labels_dict = {
            'task': 'user_identification',
            'user_labels': user_labels,
            'user_ids': user_ids,
            'trajectory_ids': trajectory_ids,
            'num_users': len(unique_users),
            'user_to_label': user_to_label,
            'label_to_user': {v: k for k, v in user_to_label.items()}
        }
        
        self.evaluation_labels['user_identification'] = labels_dict
        
        return labels_dict
    
    def extract_all_evaluation_labels(
        self,
        data_type: str = 'trajectory',
        geohash_precision: int = 7,
        uid_column: str = 'uid',
        timestamp_column: str = 'time',
        prefix_ratio: float = 0.8
    ) -> Dict:
        """
        Extract all evaluation labels at once.
        
        Args:
            data_type: 'trajectory' or 'voyages'
            geohash_precision: Geohash precision for location-based tasks
            uid_column: Name of user/trajectory ID column
            timestamp_column: Name of timestamp column
            prefix_ratio: Ratio for next location prediction
            
        Returns:
            Dictionary with all labels
        """
        print("\n" + "="*70)
        print("EXTRACTING ALL EVALUATION LABELS")
        print("="*70)
        
        # Extract all labels
        self.extract_destination_labels_geohash(
            data_type=data_type,
            geohash_precision=geohash_precision,
            uid_column=uid_column
        )
        
        self.extract_time_of_arrival_labels(
            data_type=data_type,
            uid_column=uid_column,
            timestamp_column=timestamp_column,
            normalize=True
        )
        
        # self.extract_next_location_labels_geohash(
        #     data_type=data_type,
        #     geohash_precision=geohash_precision,
        #     uid_column=uid_column,
        #     prefix_ratio=prefix_ratio
        # )
        
        self.extract_trip_purpose_labels(
            data_type=data_type,
            uid_column=uid_column,
            timestamp_column=timestamp_column
        )
        
        self.extract_user_identification_labels(
            data_type=data_type,
            uid_column=uid_column
        )
        
        print("\n" + "="*70)
        print("ALL LABELS EXTRACTED SUCCESSFULLY")
        print("="*70)
        print(f"\nAvailable labels: {list(self.evaluation_labels.keys())}")
        
        return self.evaluation_labels
    
    def get_evaluation_labels(self, task: str = None) -> Dict:
        """
        Retrieve extracted evaluation labels.
        
        Args:
            task: Specific task to retrieve, or None for all labels
            
        Returns:
            Dictionary of labels
        """
        if task is None:
            return self.evaluation_labels
        else:
            if task not in self.evaluation_labels:
                raise ValueError(f"Labels for task '{task}' not found. "
                               f"Available: {list(self.evaluation_labels.keys())}")
            return self.evaluation_labels[task]


## TEST

if __name__ == "__main__":

    tajectory_path = "~/nyc.parquet"
    data = Data(trajectory_data_path = tajectory_path)
    data.load_trajectory_data()

    # # Extract all evaluation labels
    labels = data.extract_all_evaluation_labels(
        data_type='trajectory',
        geohash_precision=7,  # ~153m resolution
        uid_column='user',
        timestamp_column='time',
        prefix_ratio=0.8
    )
    with open('Data/Labels/evaluation_labels.pkl', 'wb') as f: 
        pickle.dump(labels, f)

    # data.osm_data_from_trajectory()
    # trajectory_data_split = np.array_split(data.raw_trajectory_data,10)
    # for s, trajectory_data in enumerate(trajectory_data_split):
    #     tokens = trajectory_tokenizer(data.raw_trajectory_data, data.G, data.pois_points, data.landuse)
    #     with open('time_tokens_split'+str(s)+'.pkl','wb') as f:
    #         pickle.dump(tokens, f)