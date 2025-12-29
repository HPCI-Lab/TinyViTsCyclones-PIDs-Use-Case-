import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import pandas as pd
import numpy as np

from consts import DATA_PATH


class Era5CycloneDataset(Dataset):
    def __init__(
        self,
        split="train",
        patch_size=32,
        variable="u10",
    ):
        self.split = split
        self.patch_size = patch_size

        # ---------------- ERA5 ----------------
        self.era5_path = os.path.join(DATA_PATH, "data.grib")
        self.era5 = xr.open_dataset(self.era5_path, engine="cfgrib")

        self.data = self.era5[variable]  # (time, lat, lon)

        self.MAX = self.data.max().values
        self.MIN = self.data.min().values

        self.time = self.data["time"].values
        self.lat = self.data["latitude"].values
        self.lon = self.data["longitude"].values

        self.time_dim = len(self.time)
        self.lat_dim = len(self.lat)
        self.lon_dim = len(self.lon)

        self.n_lat_patches = self.lat_dim // self.patch_size
        self.n_lon_patches = self.lon_dim // self.patch_size
        self.patches_per_time = self.n_lat_patches * self.n_lon_patches

        # ---------------- IBTrACS ----------------

        self.ibtracs = pd.read_csv(
            "./TinyViTsCyclones-PIDs-Use-Case-/data/ibtracs.since1980.list.v04r01.csv",
            parse_dates=["ISO_TIME"],
            usecols=["ISO_TIME", "LAT", "LON"],
        )

        # Normalize longitude if needed
        if self.lon.min() >= 0:
            self.ibtracs["LON"] = self.ibtracs["LON"] % 360

        # Build fast lookup: time → list of (lat_idx, lon_idx)
        self.cyclone_lookup = self._build_cyclone_lookup()

    # -------------------------------------------------

    def _build_cyclone_lookup(self):
        """
        Map cyclone center positions to ERA5 grid indices
        """
        lookup = {}

        for _, row in self.ibtracs.iterrows():
            t = np.datetime64(row["ISO_TIME"])

            if t not in self.time:
                continue

            lat_idx = np.argmin(np.abs(self.lat - row["LAT"]))
            lon_idx = np.argmin(np.abs(self.lon - row["LON"]))

            lookup.setdefault(t, []).append((lat_idx, lon_idx))

        return lookup

    # -------------------------------------------------

    def __len__(self):
        return self.time_dim * self.patches_per_time

    # -------------------------------------------------

    def __getitem__(self, idx):
        # Map flat index → (time, patch)
        t_idx = idx // self.patches_per_time
        patch_idx = idx % self.patches_per_time

        lat_patch = patch_idx // self.n_lon_patches
        lon_patch = patch_idx % self.n_lon_patches

        lat_start = lat_patch * self.patch_size
        lon_start = lon_patch * self.patch_size

        lat_end = lat_start + self.patch_size
        lon_end = lon_start + self.patch_size

        # ERA5 patch
        patch = self.data.isel(
            time=t_idx,
            latitude=slice(lat_start, lat_end),
            longitude=slice(lon_start, lon_end),
        ).values

        x = (torch.tensor(patch, dtype=torch.float32) - self.MIN) / (self.MAX - self.MIN)

        # ---------------- Cyclone label ----------------
        y = 0
        t = self.time[t_idx]

        if t in self.cyclone_lookup:
            for lat_c, lon_c in self.cyclone_lookup[t]:
                if (
                    lat_start <= lat_c < lat_end
                    and lon_start <= lon_c < lon_end
                ):
                    y = 1
                    break

        y = torch.tensor(y, dtype=torch.float32)

        return x, y
