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
            # try:
            trajectoy_token = np.stack(single_trajectory_token, axis=-1)
            user_token.append(trajectoy_token)
            # print(trajectoy_token.shape)
        total_tokens.append(user_token)
    return total_tokens


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