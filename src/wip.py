import polars as pl
from pathlib import Path

x = Path("/media/Data/Experiments/TestMice/666/2025-03-10-13-31-19-813581/raw_data/behavior_data/valve_data.feather")
data = pl.read_ipc(source=x, use_pyarrow=True)
print(data)