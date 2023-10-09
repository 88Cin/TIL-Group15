import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def load_single_data(i):
    data = zarr.open('.\dataset\CarFollowing/trainHH.zarr', mode='a')
    start, end = data.index_range[i]

    size_lead = 4.85  # this is for AV
    size_lead = data.lead_size[i]  # this is for HV
    size_follow = data.follow_size[i]

    # get timestamps
    timestamps = data.timestamp[start:end]
    # get position, speed, and acceleration

    x_lead = data.lead_centroid[start:end]
    v_lead = data.lead_velocity[start:end]
    a_lead = data.lead_acceleration[start:end]
    id = np.ones((1, x_lead.shape[0])) * i
    x_follow = data.follow_centroid[start:end]
    v_follow = data.follow_velocity[start:end]
    a_follow = data.follow_acceleration[start:end]
    array = np.vstack((id, x_lead, v_lead, a_lead, x_follow, v_follow, a_follow))
    return array.T


def load_multi_data(i):
    for x in range(i + 1):
        single_data = load_single_data(x)
        if x == 0:
            multi_data = single_data
        else:
            multi_data = np.vstack([multi_data, single_data])
    return multi_data


np.savetxt("site.csv", load_multi_data(1000), delimiter=",")