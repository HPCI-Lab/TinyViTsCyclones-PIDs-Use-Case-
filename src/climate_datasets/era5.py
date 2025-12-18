import os
import torch
from torch.utils.data import Dataset
import xarray as xr

from consts import DATA_PATH


class Era5Dataset(Dataset):
    def __init__(self, split="train", patch_size=32, variable="u10"):
        self.split = split
        self.patch_size = patch_size

        self.era5_path = os.path.join(DATA_PATH, "data.grib")
        self.era5 = xr.open_dataset(self.era5_path, engine="cfgrib")

        # (time, lat, lon)
        self.data = self.era5[variable]
        self.MAX = self.data.max().values
        self.MIN = self.data.min().values

        self.time_dim = self.data.sizes["time"]
        self.lat_dim = self.data.sizes["latitude"]
        self.lon_dim = self.data.sizes["longitude"]

        # Number of patches per dimension (no overlap, drop remainder)
        self.n_lat_patches = self.lat_dim // self.patch_size
        self.n_lon_patches = self.lon_dim // self.patch_size
        self.patches_per_time = self.n_lat_patches * self.n_lon_patches

    def __len__(self):
        return self.time_dim * self.patches_per_time

    def __getitem__(self, idx):
        # --- map flat idx → (time, lat_patch, lon_patch) ---
        t = idx // self.patches_per_time
        patch_idx = idx % self.patches_per_time

        lat_patch = patch_idx // self.n_lon_patches
        lon_patch = patch_idx % self.n_lon_patches

        lat_start = lat_patch * self.patch_size
        lon_start = lon_patch * self.patch_size

        patch = self.data.isel(
            time=t,
            latitude=slice(lat_start, lat_start + self.patch_size),
            longitude=slice(lon_start, lon_start + self.patch_size),
        ).values

        p = (torch.tensor(patch, dtype=torch.float32) - self.MIN) / (self.MAX - self.MIN)
        return p, p
