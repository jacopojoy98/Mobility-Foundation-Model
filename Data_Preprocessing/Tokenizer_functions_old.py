import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import geopandas as gpd
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
import pickle
from osmnx._errors import InsufficientResponseError
import os
import json
import shutil
import tqdm
import pickle
# ============================================================
# HELPER FUNCTIONS
# ============================================================
def haversine_dist_approx(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in meters."""
    return np.sqrt(((lat1-lat2)*111320)**2 + ((lon1-lon2)*111320)**2)
    

def haversine_dist(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in meters."""
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def bearing(p1, p2):
    """Bearing in radians."""
    lat1, lon1, lat2, lon2 = map(math.radians, [p1["lat"], p1["lon"], p2["lat"], p2["lon"]])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.atan2(y, x)


# ============================================================
# OSM DATA FETCHING
# ============================================================

def safe_features_from_point(center_point, tags, dist):
    try:
        gdf = ox.features_from_point(center_point, tags=tags, dist=dist)
        return gdf
    except InsufficientResponseError:
        print(f"⚠️  No features found for tags {tags}. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(geometry=[])


def load_osm_data(center_point, dist=2000, cache_dir="osm_cache", cache_name=None):
    """
    Load OSM data (road network, intersections, POIs, land-use polygons) for an area.
    Uses on-disk caching to avoid re-downloading the same data.
    """

    # ------------------------------------------------------------
    # 1. Compute cache path
    # ------------------------------------------------------------
    lat, lon = center_point
    if cache_name is None:
        cache_name = f"osm_{round(lat, 4)}_{round(lon, 4)}_{dist}m"

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    graph_file = f"{cache_path}_graph.graphml"
    edges_file = f"{cache_path}_edges.gpkg"
    nodes_file = f"{cache_path}_nodes.gpkg"
    pois_file = f"{cache_path}_pois.gpkg"
    landuse_file = f"{cache_path}_landuse.gpkg"

    # ------------------------------------------------------------
    # 2. Try to load from cache
    # ------------------------------------------------------------
    if all(os.path.exists(f) for f in [graph_file, edges_file, nodes_file, pois_file, landuse_file]):
        print(f"📦 Loading OSM data from cache: {cache_path}")
        G = ox.load_graphml(graph_file)
        gdf_edges = gpd.read_file(edges_file)
        gdf_nodes = gpd.read_file(nodes_file)
        pois = gpd.read_file(pois_file)
        landuse = gpd.read_file(landuse_file)
        intersections = [(n.y, n.x) for n in gdf_nodes.geometry]
        pois_points = pois[pois.geometry.type == "Point"].geometry
        return G, gdf_edges, gdf_nodes, intersections, pois_points, landuse

    # ------------------------------------------------------------
    # 3. Otherwise, download from OSM
    # ------------------------------------------------------------
    print(f"🌐 Downloading OSM data for {center_point} (dist={dist} m)...")
    def try_load(dist, network_type):
        print(f"Trying OSM download: dist={dist}, type={network_type}")
        return ox.graph_from_point(center_point, dist=dist, network_type=network_type)
    try:
        G = try_load(dist, "drive")
    except ValueError:
        print("No 'drive' network found, retrying with larger distance...")
        try:
            G = try_load(dist * 2, "drive")
        except ValueError:
            print("Still no results, switching to 'all' network type...")
            G = try_load(dist * 2, "all")

    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    intersections = [(n.y, n.x) for n in gdf_nodes.geometry]

    # --- Safe feature loading ---
    def safe_features_from_point(tags):
        try:
            gdf = ox.features_from_point(center_point, tags=tags, dist=dist)
            return gdf
        except InsufficientResponseError:
            print(f"⚠️  No features found for tags {tags}. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(geometry=[])

    # POIs and landuse
    pois = safe_features_from_point({'amenity': True, 'shop': True, 'tourism': True, 'leisure': True})
    landuse = safe_features_from_point({'landuse': True})

    # --- Compute centrality ---
    print("Computing network centrality (this may take a while for large graphs)...")
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = nx.Graph(G)
    centrality_dict = nx.betweenness_centrality(G_undirected, k=500, normalized=True)
    nx.set_node_attributes(G, centrality_dict, "centrality")

    # ------------------------------------------------------------
    # 4. Save to cache for reuse
    # ------------------------------------------------------------
    print(f"💾 Saving OSM data to cache: {cache_path}")
    ox.save_graphml(G, graph_file)
    gdf_edges.to_file(edges_file, driver="GPKG")
    gdf_nodes.to_file(nodes_file, driver="GPKG")
    pois.to_file(pois_file, driver="GPKG")
    landuse.to_file(landuse_file, driver="GPKG")

    pois_points = pois[pois.geometry.type == "Point"].geometry
    return G, gdf_edges, gdf_nodes, intersections, pois_points, landuse



# ============================================================
# FEATURE COMPUTATION FUNCTIONS
# ============================================================

def compute_speed(trajectory):
    speeds = [0.0]
    for i in range(1, len(trajectory)):
        p1, p2 = trajectory[i - 1], trajectory[i]
        dt = (p2["time"] - p1["time"]).total_seconds()
        if dt <= 0:
            speeds.append(0.0)
            continue
        dist = haversine_dist_approx(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        speeds.append(dist / dt)
    return speeds


def compute_turn_angle(trajectory):
    bearings = [0.0]
    for i in range(1, len(trajectory)):
        bearings.append(bearing(trajectory[i-1], trajectory[i]))
    turn_angles = [0.0]
    for i in range(1, len(bearings)):
        dtheta = bearings[i] - bearings[i-1]
        while dtheta > math.pi: dtheta -= 2 * math.pi
        while dtheta < -math.pi: dtheta += 2 * math.pi
        turn_angles.append(dtheta)
    return turn_angles


def compute_time_gap(trajectory):
    return [0.0] + [(trajectory[i]["time"] - trajectory[i-1]["time"]).total_seconds() for i in range(1, len(trajectory))]


def map_match_point_to_road(G, point):
    nearest_edge = ox.distance.nearest_edges(G, point["lon"], point["lat"])
    u, v, key = nearest_edge
    edge_data = G[u][v][key]
    return edge_data

def point_road_type(point, G):
    rt = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service"]
    try:
        edge_data = map_match_point_to_road(G, point)
        road_type = edge_data.get("highway", "unknown")
        if isinstance(road_type, list):
            road_type = road_type[0]
    except Exception:
        road_type = "unknown"
    road_type = rt.index(road_type) if road_type in rt else -1
    return road_type

def compute_road_type(trajectory, G):
    road_types = []
    for p in trajectory:
        road_type = point_road_type(p, G)
        road_types.append(road_type)
    return road_types

## dist_to_intersection
def point_distance_to_intersection(point, intersections):
    d_min = min(geodesic((point["lat"], point["lon"]), inter).meters for inter in intersections)
    return d_min

def compute_distance_to_intersection(trajectory, intersections):
    distances = []
    for p in trajectory:
        d_min = point_distance_to_intersection(p, intersections)
        distances.append(d_min)
    return distances

## poi_density
def point_local_poi_density(point, pois_points, radius=100):
    count = 0 
    for (x,y) in pois_points:
        if ((x-point["lat"])*111320)**2 + ((y-point["lon"])*111320)**2 <= radius**2:
            count +=1
    area = math.pi * radius ** 2
    return count / area

def compute_local_poi_density(trajectory, pois_points, radius=100):
    densities = []
    for p in trajectory:
        density = point_local_poi_density(p, pois_points, radius)
        densities.append(density)
    return densities

## land_use_diversity
def point_land_use_diversity(point, landuse, radius=100):
    buffer = Point(point["lon"], point["lat"]).buffer(radius / 111320)
    nearby = landuse[landuse.intersects(buffer)]
    if nearby.empty or "landuse" not in nearby.columns:
        return 0.0
    values, counts = np.unique(nearby["landuse"].astype(str), return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)

def compute_land_use_diversity(trajectory, landuse, radius=100):
    entropies = []
    for p in trajectory:
        entropy = point_land_use_diversity(p, landuse, radius)
        entropies.append(entropy)
    return entropies


## node_centrality
def point_network_centrality(G, point):
    try:
        u, v, key = ox.distance.nearest_edges(G, point["lon"], point["lat"])
        node_centrality = (G.nodes[u].get("centrality", 0.0) + G.nodes[v].get("centrality", 0.0)) / 2
    except Exception:
        node_centrality = 0.0
    return node_centrality

def compute_network_centrality(trajectory, G):
    centr = []
    for p in trajectory:
        node_centrality = point_network_centrality(G, p)
        centr.append(node_centrality)
    return centr
