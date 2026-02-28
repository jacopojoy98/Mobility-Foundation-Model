import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import datetime
import Tokenizer_functions as Tf
def voyages_tokenizer(raw_voyages_data: pd.DataFrame):
    return 0


def load_osm_data(center_point: tuple, radius: int, cache_dir: str = "osm_cache"):
    lat, lon = center_point

def find_trajectory_center(raw_trajectory_data: pd.DataFrame) -> tuple :
    return 0

def trajectory_tokenizer(raw_trajectory_data: gpd.GeoDataFrame, G, pois, landuse):
    total_tokens = []
    for user, user_trajectory_data in tqdm(raw_trajectory_data.groupby(['user'])):
        user_token=[]
        for traj_id, single_trajectory_data in user_trajectory_data.groupby(['traj_id']):
            single_trajectory_token = []
            speed = Tf.speed(single_trajectory_data)#  [0-30,30-70,70-100,100+]
            single_trajectory_token.append(speed)
            turn_angle=Tf.turn_angle(single_trajectory_data)# [0-45,45-90,90-135,135-180]
            single_trajectory_token.append(turn_angle)
            tod = Tf.tod(single_trajectory_data)# [7-13,13-17,17-22,22-7]
            single_trajectory_token.append(tod)
            weekday = Tf.weekday(single_trajectory_data)
            single_trajectory_token.append(weekday)
            road_type = Tf.road_type(single_trajectory_data, G)
            single_trajectory_token.append(road_type)
            poi_density = Tf.poi_density(single_trajectory_data, pois)
            single_trajectory_token.append(poi_density)
            landuse_diversity = Tf.landuse_diversity(single_trajectory_data, landuse) 
            single_trajectory_token.append(landuse_diversity)
            network_centrality = Tf.network_centrality(single_trajectory_data, G)
            single_trajectory_token.append(network_centrality)
            # print(speed.shape, turn_angle.shape, tod.shape, weekday.shape, road_type.shape, poi_density.shape, landuse_diversity.shape, network_centrality.shape)
            # try:
            trajectoy_token = np.stack(single_trajectory_token, axis=-1)
            user_token.append(trajectoy_token)
            # print(trajectoy_token.shape)
        total_tokens.append(user_token)
    return total_tokens

def visit_tokenizer_from_trajectory_tokens(trajectory_tokens: list):
    '''
    visit_tokens = [
        user_n_visit_tokens = [
            visit_n_token = [
                weekday, start_time_of_day, end_time_of_day, start_road_type, end_road_type, 
                start_poi_density, end_poi_density, start_network_centrality, end_network_centrality]

        ]
    ]
    '''    
    visit_tokens = []
    for user_trajectories in trajectory_tokens:
        user_visit_tokens = []
        for traj_token in user_trajectories:
            visit_token = [] 
            weekday = traj_token[0, 3]  # Assuming weekday is the 4th feature
            start_time_of_day = traj_token[0, 2]  # Assuming time of day is the 3rd feature
            end_time_of_day = traj_token[-1, 2]  # Assuming time of day is the 3rd feature
            start_road_type = traj_token[0, 4]  # Assuming road type is the 5th feature
            end_road_type = traj_token[-1, 4]  # Assuming road type is the 5th feature
            start_poi_density = traj_token[0, 5]  # Assuming poi density is the 6th feature
            end_poi_density = traj_token[-1, 5]  # Assuming poi density is the 6th feature
            start_time_of_day = traj_token[0, 2]  # Assuming time of day is the 3rd feature
            start_network_centrality = traj_token[0, 7]  # Assuming network centrality is the 8th feature
            end_network_centrality = traj_token[-1, 7]  # Assuming network centrality is the 8th feature

            
            visit_token.append(weekday)  # Replace weekday with actual value
            visit_token.append(start_time_of_day)  # Replace time of day with start time of day
            visit_token.append(end_time_of_day)  # Replace time of day with end time of day
            visit_token.append(start_road_type)  # Replace road type with start road type
            visit_token.append(end_road_type)  # Replace road type with end road type
            visit_token.append(start_poi_density)  # Replace poi density with start poi density
            visit_token.append(end_poi_density)  # Replace poi density with end poi density
            visit_token.append(start_network_centrality)  # Replace network centrality with start network centrality
            visit_token.append(end_network_centrality)  # Replace network centrality with end network centrality

            user_visit_tokens.append(visit_token) 
        visit_tokens.append(user_visit_tokens)
    return visit_tokens

def change_type_tokens(trajectory_tokens: list):
    total_tokens = []
    for user in trajectory_tokens:
        user_token = []
        for traj in user:
            traj_token = []
            for point in traj:
                point_token = []
                for feature in point:
                    point_token.append(float(feature))  # Convert feature to float
                traj_token.append(point_token)
            user_token.append(traj_token)
        total_tokens.append(user_token)
    return total_tokens



if __name__ == "__main__":
    complete = []
    for i in range(10):
        with open(f"trajectory_tokens_split_{i}.pkl", "rb") as f:
            trajectory_token = pickle.load(f)
        for user in trajectory_token:
            complete.append(user)
    
    with open(f"complete_trajectory.pkl", "wb") as f:
        pickle.dump(complete, f)
# def trajectory_downstream(raw_trajectory_data: gpd.GeoDataFrame):
#     for user, user_trajectory_data in raw_trajectory_data.groupby(['user']):
#         for traj_id, single_trajectroy_data in user_trajectory_data.groupby(['traj_id']):
#             _destination = destination(single_trajectroy_data)
#             _time_of_arrival = time_of_arrival(single_trajectroy_data)

    '''
    
    ttt= []
    for _, row in dataframe.iterrows():
        p = {"lat":row["latitude"], "lon":row["longitude"], "time": datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S"), "trip" : j }
        ttt.append(p)
        
    features_traj = {
        "trip" : [p["trip"] for p in ttt],
        "speed": speed(ttt),
        "turn_angle": compute_turn_angle(ttt),
        "time": [p["time"] for p in ttt],
        "minutes": [p["time"].timestamp() / 60 for p in ttt],
        "tod": [p["time"].timestamp() % (24 * 60 * 60) / (24 * 60 * 60) for p in ttt],
        "weekday": [p["time"].weekday() for p in ttt],
        "time_gap": compute_time_gap(ttt),
        # "road_type": compute_road_type(ttt, G),
        # "dist_to_intersection": compute_distance_to_intersection(ttt, intersections),
        # "poi_density": compute_local_poi_density(ttt, pois_points),
        # "landuse_diversity": compute_land_use_diversity(ttt, landuse),
        # "network_centrality": compute_network_centrality(ttt, G),
        }
        
    for key in features_traj:
        if key not in total_features.keys():
            total_features[key] = []
        total_features[key].extend(features_traj[key])
                
    dataframe = pd.DataFrame.from_dict(total_features)
    dataframe.to_csv("trajectories_features.csv", index=False)


    return 0
    '''