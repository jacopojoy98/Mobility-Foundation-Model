"""
Label Extraction for Evaluation Tasks
======================================

This script extracts labels from raw trajectory data for downstream evaluation tasks.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
from collections import Counter
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta


class LabelExtractor:
    """
    Extracts various labels from trajectory data for evaluation.
    """
    
    @staticmethod
    def extract_destination(trajectory_sequence):
        """
        Extract destination (final location) from trajectory.
        
        Args:
            trajectory_sequence: Array of shape (seq_len, token_dim)
                                First 2 dimensions should be spatial (lat, lon or x, y)
        
        Returns:
            destination: Spatial coordinates of final location
        """
        # Return last location
        return trajectory_sequence[-1, :2]  # First 2 dims are spatial
    
    @staticmethod
    def extract_time_of_arrival(trajectory_sequence, time_dim=2):
        """
        Extract time of arrival from trajectory.
        
        Args:
            trajectory_sequence: Array of shape (seq_len, token_dim)
            time_dim: Index of time dimension in token vector
        
        Returns:
            arrival_time: Time at final location
        """
        # Return time at last location
        return trajectory_sequence[-1, time_dim]
    
    @staticmethod
    def extract_travel_duration(trajectory_sequence, time_dim=2):
        """
        Extract total travel duration.
        
        Args:
            trajectory_sequence: Array of shape (seq_len, token_dim)
            time_dim: Index of time dimension in token vector
        
        Returns:
            duration: Travel duration (end_time - start_time)
        """
        start_time = trajectory_sequence[0, time_dim]
        end_time = trajectory_sequence[-1, time_dim]
        return end_time - start_time
    
    @staticmethod
    def extract_next_location(trajectory_sequence, prefix_length=None):
        """
        Extract next location given a prefix of the trajectory.
        
        Args:
            trajectory_sequence: Array of shape (seq_len, token_dim)
            prefix_length: Length of prefix (if None, use 80% of trajectory)
        
        Returns:
            prefix: Prefix trajectory
            next_location: Next location coordinates
        """
        if prefix_length is None:
            prefix_length = int(len(trajectory_sequence) * 0.8)
        
        prefix_length = max(1, min(prefix_length, len(trajectory_sequence) - 1))
        
        prefix = trajectory_sequence[:prefix_length]
        next_location = trajectory_sequence[prefix_length, :2]
        
        return prefix, next_location
    
    @staticmethod
    def cluster_locations(all_trajectories, eps=0.01, min_samples=5):
        """
        Cluster locations across all trajectories to create discrete location labels.
        Useful for destination prediction and next location prediction.
        
        Args:
            all_trajectories: List of trajectory arrays
            eps: DBSCAN epsilon parameter (distance threshold)
            min_samples: DBSCAN min_samples parameter
        
        Returns:
            location_labels: Dictionary mapping trajectory_idx -> location_cluster_id
            clusterer: Fitted DBSCAN clusterer
        """
        # Extract all locations (first 2 dimensions)
        all_locations = []
        trajectory_indices = []
        
        for traj_idx, traj in enumerate(all_trajectories):
            for loc in traj[:, :2]:
                all_locations.append(loc)
                trajectory_indices.append(traj_idx)
        
        all_locations = np.array(all_locations)
        
        # Cluster locations
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(all_locations)
        
        # Map back to trajectories
        location_labels = {}
        for i, (traj_idx, cluster_id) in enumerate(zip(trajectory_indices, cluster_labels)):
            if traj_idx not in location_labels:
                location_labels[traj_idx] = []
            location_labels[traj_idx].append(cluster_id)
        
        # Get destination cluster for each trajectory
        destination_clusters = {}
        for traj_idx, clusters in location_labels.items():
            destination_clusters[traj_idx] = clusters[-1]  # Last location cluster
        
        return destination_clusters, clusterer
    
    @staticmethod
    def infer_trip_purpose(trajectory_sequence, time_dim=2, day_dim=None):
        """
        Infer trip purpose from trajectory characteristics.
        
        Simple heuristic:
        - Work: weekday, morning (7-9am) or evening (5-7pm)
        - Home: late evening/night (10pm-6am)
        - Leisure: weekend or weekday midday/evening
        
        Args:
            trajectory_sequence: Array of shape (seq_len, token_dim)
            time_dim: Index of time dimension (0-1 normalized)
            day_dim: Index of day dimension (0-1 normalized, optional)
        
        Returns:
            purpose: Integer label (0=work, 1=home, 2=leisure, 3=other)
        """
        # Get time at destination
        time_normalized = trajectory_sequence[-1, time_dim]
        
        # Convert normalized time to hour (assuming 0-1 maps to 0-24 hours)
        hour = time_normalized * 24
        
        # Get day if available
        is_weekend = False
        if day_dim is not None:
            day_normalized = trajectory_sequence[-1, day_dim]
            day_of_week = int(day_normalized * 7)
            is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6
        
        # Infer purpose
        if not is_weekend:
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 0  # Work
            elif 22 <= hour or hour <= 6:
                return 1  # Home
            else:
                return 2  # Leisure
        else:
            if 22 <= hour or hour <= 6:
                return 1  # Home
            else:
                return 2  # Leisure
    
    @staticmethod
    def generate_evaluation_labels(
        trajectories: List[np.ndarray],
        user_ids: List[int] = None,
        time_dim: int = 2,
        day_dim: int = None,
        destination_eps: float = 0.01,
        prefix_ratio: float = 0.8
    ) -> Dict:
        """
        Generate all evaluation labels from trajectory data.
        
        Args:
            trajectories: List of trajectory arrays (seq_len, token_dim)
            user_ids: Optional list of user IDs for each trajectory
            time_dim: Index of time dimension in token vector
            day_dim: Index of day dimension in token vector (optional)
            destination_eps: DBSCAN epsilon for clustering destinations
            prefix_ratio: Ratio of trajectory to use as prefix for next location
        
        Returns:
            labels: Dictionary containing all extracted labels
        """
        print("Extracting evaluation labels from trajectories...")
        print(f"  Number of trajectories: {len(trajectories)}")
        
        labels = {}
        
        # 1. Destination labels (clustered)
        print("\n1. Extracting destination labels...")
        destination_coords = [traj[-1, :2] for traj in trajectories]
        destination_coords_array = np.array(destination_coords)
        
        # Cluster destinations
        clusterer = DBSCAN(eps=destination_eps, min_samples=5)
        destination_labels = clusterer.fit_predict(destination_coords_array)
        
        # Handle noise points (-1 labels) by mapping to separate classes
        unique_labels = set(destination_labels)
        if -1 in unique_labels:
            # Map -1 to new unique labels
            max_label = max([l for l in unique_labels if l != -1])
            noise_mapper = {}
            noise_counter = max_label + 1
            
            new_destination_labels = []
            for label, coords in zip(destination_labels, destination_coords):
                if label == -1:
                    coord_key = tuple(coords)
                    if coord_key not in noise_mapper:
                        noise_mapper[coord_key] = noise_counter
                        noise_counter += 1
                    new_destination_labels.append(noise_mapper[coord_key])
                else:
                    new_destination_labels.append(label)
            
            destination_labels = np.array(new_destination_labels)
        
        num_destination_classes = len(set(destination_labels))
        print(f"   Number of destination clusters: {num_destination_classes}")
        
        labels['destinations'] = destination_labels
        labels['num_destinations'] = num_destination_classes
        
        # 2. Time of arrival labels
        print("\n2. Extracting time of arrival labels...")
        arrival_times = np.array([traj[-1, time_dim] for traj in trajectories])
        print(f"   Time range: [{arrival_times.min():.3f}, {arrival_times.max():.3f}]")
        
        labels['arrival_times'] = arrival_times
        
        # 3. Travel duration labels
        print("\n3. Extracting travel duration labels...")
        durations = []
        for traj in trajectories:
            duration = LabelExtractor.extract_travel_duration(traj, time_dim)
            durations.append(duration)
        durations = np.array(durations)
        print(f"   Duration range: [{durations.min():.3f}, {durations.max():.3f}]")
        
        labels['durations'] = durations
        
        # 4. Next location labels
        print("\n4. Extracting next location labels...")
        prefix_trajectories = []
        next_locations = []
        
        for traj in trajectories:
            prefix_length = int(len(traj) * prefix_ratio)
            prefix_length = max(1, min(prefix_length, len(traj) - 1))
            
            prefix = traj[:prefix_length]
            next_loc = traj[prefix_length, :2]
            
            prefix_trajectories.append(prefix)
            next_locations.append(next_loc)
        
        # Cluster next locations
        next_locations_array = np.array(next_locations)
        next_loc_clusterer = DBSCAN(eps=destination_eps, min_samples=5)
        next_location_labels = next_loc_clusterer.fit_predict(next_locations_array)
        
        # Handle noise points
        unique_labels = set(next_location_labels)
        if -1 in unique_labels:
            max_label = max([l for l in unique_labels if l != -1])
            noise_mapper = {}
            noise_counter = max_label + 1
            
            new_next_labels = []
            for label, coords in zip(next_location_labels, next_locations):
                if label == -1:
                    coord_key = tuple(coords)
                    if coord_key not in noise_mapper:
                        noise_mapper[coord_key] = noise_counter
                        noise_counter += 1
                    new_next_labels.append(noise_mapper[coord_key])
                else:
                    new_next_labels.append(label)
            
            next_location_labels = np.array(new_next_labels)
        
        num_next_location_classes = len(set(next_location_labels))
        print(f"   Number of next location clusters: {num_next_location_classes}")
        
        labels['prefix_trajectories'] = prefix_trajectories
        labels['next_locations'] = next_location_labels
        labels['num_next_locations'] = num_next_location_classes
        
        # 5. Trip purpose labels
        print("\n5. Inferring trip purpose labels...")
        purposes = []
        for traj in trajectories:
            purpose = LabelExtractor.infer_trip_purpose(traj, time_dim, day_dim)
            purposes.append(purpose)
        purposes = np.array(purposes)
        
        purpose_counts = Counter(purposes)
        print(f"   Purpose distribution:")
        purpose_names = {0: 'Work', 1: 'Home', 2: 'Leisure', 3: 'Other'}
        for purpose_id, count in sorted(purpose_counts.items()):
            print(f"     {purpose_names.get(purpose_id, 'Unknown')}: {count} ({count/len(purposes)*100:.1f}%)")
        
        labels['purposes'] = purposes
        labels['num_purposes'] = len(set(purposes))
        
        # 6. User identification labels
        if user_ids is not None:
            print("\n6. Processing user IDs...")
            user_ids = np.array(user_ids)
            unique_users = len(set(user_ids))
            print(f"   Number of unique users: {unique_users}")
            
            labels['user_ids'] = user_ids
            labels['num_users'] = unique_users
        else:
            print("\n6. No user IDs provided, skipping user identification labels")
        
        print("\n" + "="*70)
        print("Label extraction complete!")
        print("="*70)
        
        return labels
    
    @staticmethod
    def split_data(
        trajectories: List[np.ndarray],
        labels: Dict,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[Dict, Dict]:
        """
        Split trajectories and labels into train/val sets.
        
        Args:
            trajectories: List of trajectory arrays
            labels: Dictionary of labels from generate_evaluation_labels
            train_ratio: Ratio of data to use for training
            random_seed: Random seed for reproducibility
        
        Returns:
            train_data: Dictionary with training trajectories and labels
            val_data: Dictionary with validation trajectories and labels
        """
        np.random.seed(random_seed)
        
        num_samples = len(trajectories)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        split_point = int(num_samples * train_ratio)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        # Split trajectories
        train_trajectories = [trajectories[i] for i in train_indices]
        val_trajectories = [trajectories[i] for i in val_indices]
        
        # Split labels
        train_labels = {}
        val_labels = {}
        
        for key, value in labels.items():
            if key.startswith('num_'):
                # Copy metadata
                train_labels[key] = value
                val_labels[key] = value
            elif key == 'prefix_trajectories':
                # List of arrays
                train_labels[key] = [value[i] for i in train_indices]
                val_labels[key] = [value[i] for i in val_indices]
            elif isinstance(value, np.ndarray):
                # Numpy arrays
                train_labels[key] = value[train_indices]
                val_labels[key] = value[val_indices]
        
        train_data = {
            'trajectories': train_trajectories,
            'labels': labels
        }
        
        val_data = {
            'trajectories': val_trajectories,
            'labels': labels
        }
        
        print(f"\nData split:")
        print(f"  Training samples: {len(train_trajectories)}")
        print(f"  Validation samples: {len(val_trajectories)}")
        
        return train_data, val_data


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Label Extraction Example")
    print("="*70)
    
    # Generate example trajectories
    num_trajectories = 1000
    num_users = 50
    
    print(f"\nGenerating {num_trajectories} example trajectories...")
    
    trajectories = []
    user_ids = []
    
    for i in range(num_trajectories):
        # Random trajectory length
        seq_len = np.random.randint(20, 100)
        
        # Random token dimension 4: [lat, lon, time, day]
        trajectory = np.random.randn(seq_len, 4)
        
        # Make time monotonically increasing and normalized
        trajectory[:, 2] = np.linspace(0, 1, seq_len)
        
        # Make day consistent within trajectory
        trajectory[:, 3] = np.random.rand()
        
        trajectories.append(trajectory)
        user_ids.append(i % num_users)  # Assign users
    
    # Extract labels
    labels = LabelExtractor.generate_evaluation_labels(
        trajectories,
        user_ids=user_ids,
        time_dim=2,
        day_dim=3,
        destination_eps=0.1,
        prefix_ratio=0.8
    )
    
    # Split data
    train_data, val_data = LabelExtractor.split_data(
        trajectories,
        labels,
        train_ratio=0.8,
        random_seed=42
    )
    
    print("\n" + "="*70)
    print("Labels extracted and data split successfully!")
    print("="*70)
    print("\nAvailable labels:")
    for key in labels.keys():
        if not key.startswith('num_'):
            print(f"  - {key}")
