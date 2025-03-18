from pathlib import Path

import numpy as np
import polars as pl

from sl_experiment.module_interfaces import _interpolate_data

root = Path("/media/Data/Experiments/Template/666/2025-03-18-18-52-54-948030/raw_data/behavior_data")
x = root.joinpath("break_data.feather")
y = pl.read_ipc(x, use_pyarrow=True)
print(y)

import sys

sys.exit()

# # Load data
# distance = Path("/media/Data/Experiments/TestMice/666/2025-03-10-16-15-25-577230/raw_data/behavior_data/encoder_data.feather")
# reward = Path("/media/Data/Experiments/TestMice/666/2025-03-10-16-15-25-577230/raw_data/behavior_data/valve_data.feather")
# distance_data = pl.read_ipc(source=distance, use_pyarrow=True)
# reward_data = pl.read_ipc(source=reward, use_pyarrow=True)
#
# # Extract relevant columns
# time_d = distance_data["time_us"]
# distance_cm = distance_data["traveled_distance_cm"]
# time_r = reward_data["time_us"]
# tone_r = reward_data["dispensed_water_volume_uL"]
#
# # Identify indices where the delivered volume increases
# volume_changes = np.where(np.diff(tone_r) > 0)[0] + 1
# tone_times = time_r[volume_changes]
#
# def f(distance_cm):
#     return (distance_cm / 10.0) % 24
#
# # Interpolate reward distances
# reward_distances = _interpolate_data(time_d, distance_cm, tone_times, is_discrete=False)
# for num, distance in enumerate(reward_distances):
#     reward_distances[num] = f(distance)
#
# print(reward_distances)
# print(distance_cm[-1]/240)
