import numpy as np
import pandas as pd
import torch

def pointwise_distance(point_1: pd.Series, point_2: pd.Series):    
    return np.sqrt(((point_1['lat']-point_2['lat'])*111320)**2 + ((point_1['lon']-point_2['lon'])*111320)**2)

def speed(trajectory: pd.DataFrame):
    tmp_speed = []
    for i in range(len(trajectory)-1):
        first_point = trajectory.iloc[i]
        second_point = trajectory.iloc[i+1]
        hop_radius_meters = pointwise_distance(first_point, second_point)
        hop_time_seconds = (second_point['timestamp'] - first_point['timestamp']).seconds()
        tmp_speed.append(hop_radius_meters/hop_time_seconds)    
    return tmp_speed

def turn_angle():
    return 0