import numpy as np
import geopandas as gpd
import math
import shapely
import osmnx as ox

road_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",\
               "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street",\
                "pedestrian", "track", "bus_guideway", "escape", "raceway", "road", "busway"]

def pointwise_distance(point_1: shapely.Point, point_2: shapely.Point):    
    return np.sqrt(((point_1.x-point_2.x)*111320)**2 + ((point_1.y-point_2.y)*111320)**2)

def speed(trajectory: gpd.GeoDataFrame):
    tmp_speed = [0.0]
    for i in range(len(trajectory)-1):
        first_point = trajectory.iloc[i]
        second_point = trajectory.iloc[i+1]
        hop_radius_meters = pointwise_distance(first_point["geometry"], second_point["geometry"])
        hop_time_seconds = (second_point['time'].timestamp() - first_point['time'].timestamp())
        if hop_time_seconds == 0:
            tmp_speed.append(0)    
        else:
            tmp_speed.append(hop_radius_meters/hop_time_seconds)    
    return np.array(tmp_speed)

def bearing(p1: shapely.Point, p2: shapely.Point):
    lat1, lon1, lat2, lon2 = map(math.radians, [p1.x, p1.y, p2.x, p2.y])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.atan2(y, x)

def turn_angle(trajectory):
    bearings = [0.0]
    for i in range(1, len(trajectory)):
        bearings.append(bearing(trajectory.iloc[i-1]["geometry"], trajectory.iloc[i]["geometry"]))
    turn_angles = [0.0]
    for i in range(1, len(bearings)):
        dtheta = bearings[i] - bearings[i-1]
        while dtheta > math.pi: dtheta -= 2 * math.pi
        while dtheta < -math.pi: dtheta += 2 * math.pi
        turn_angles.append(dtheta)
    return np.array(turn_angles)

def minute(trajectory: gpd.GeoDataFrame):
    tmp_minute = []
    for i in range(len(trajectory)):
        _minute = trajectory.iloc[i]["time"].timestamp()%(60 * 60) / (60 * 60)
        tmp_minute.append(_minute)
    return np.array(tmp_minute)

def tod(trajectory: gpd.GeoDataFrame):
    tmp_tod = []
    for i in range(len(trajectory)):
        _time_of_day = trajectory.iloc[i]["time"].timestamp() % (24 * 60 * 60) / (24 * 60 * 60)
        tmp_tod.append(_time_of_day)
    return np.array(tmp_tod)

def weekday(trajectory: gpd.GeoDataFrame):
    tmp_weekday = []
    for i in range(len(trajectory)):
        _weekday = trajectory.iloc[i]["time"].weekday()
        tmp_weekday.append(_weekday)
    return np.array(tmp_weekday)

def road_type(trajectory: gpd.GeoDataFrame, G):
    tmp_road_type = []
    nearest_edges = ox.nearest_edges(G, trajectory["geometry"].x, trajectory["geometry"].y)
    for nearest_edge in nearest_edges:
        u,v,key = nearest_edge
        edge_data = G[u][v][key]
        _road_type = edge_data.get("highway")
        if type(_road_type) == list:
            _road_type = _road_type[0] 
        tmp_road_type.append(road_types.index(_road_type))
    return np.array(tmp_road_type)

def poi_density(trajectory: gpd.GeoDataFrame, pois: gpd.GeoSeries):
    tmp_poi_density = []
    for i in range(len(trajectory)):
        zone_around_point = trajectory.iloc[i]["geometry"].buffer(1e-3) #1e-3 radiants is apprx 100m
        _poi_density = sum(pois.within(zone_around_point))
        tmp_poi_density.append(_poi_density)        
    return np.array(tmp_poi_density)

def landuse_diversity(trajectory: gpd.GeoDataFrame, landuse: gpd.GeoSeries):
    tmp_landuse_diversity = []
    for i in range(len(trajectory)):
        zone_around_point = trajectory.iloc[i]["geometry"].buffer(1e-3)
        nearby = landuse[landuse.intersects(zone_around_point)]
        if nearby.empty or "landuse" not in nearby.columns:
            tmp_landuse_diversity.append(0.0)
        else:
            values, counts = np.unique(nearby["landuse"].astype(str), return_counts=True)
            probs = counts / counts.sum()
            tmp_landuse_diversity.append(-np.sum(probs * np.log(probs)))
    return np.array(tmp_landuse_diversity)

def network_centrality(trajectory: gpd.GeoDataFrame, G):
    tmp_network_centrality = []
    nodes = ox.nearest_nodes(G, trajectory["geometry"].x, trajectory["geometry"].y)
    for node in nodes:
        _network_centrality = G.nodes[node].get("centrality")
        tmp_network_centrality.append(_network_centrality)
    return np.array(tmp_network_centrality)


