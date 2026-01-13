import pandas as pd
import geopandas as gpd
import os
import tqdm
import datetime

def voyages_tokenizer(raw_voyages_data: pd.DataFrame):
    return 0


def load_osm_data(center_point: tuple, radius: int, cache_dir: str = "osm_cache"):
    lat, lon = center_point

def find_trajectory_center(raw_trajectory_data: pd.DataFrame) -> tuple :
    return 0

def trajectory_tokenizer(raw_trajectory_data: pd.DataFrame):
    
    center_point = find_trajectory_center(raw_trajectory_data)
    radius_meters = 1000
    G, edges, nodes, intersections, pois_points, landuse = load_osm_data(center_point, radius_meters)
    pois_points = [(p.y, p.x) for p in pois_points]
    dir_traj = os.path.join("Data","final_data_ny")
    for j, filename in tqdm.tqdm(enumerate(os.listdir(dir_traj)), total = 9998):
        dataframe = pd.read_csv(os.path.join(dir_traj, filename))
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