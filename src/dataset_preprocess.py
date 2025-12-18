
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import yprov4ml
import geopandas as gpd

from consts import DATA_PATH


# X -5 36
# Y 20 50

latitude = 161
longitude = 81

def main(): 
    # yprov4ml.start_run(
    #     experiment_name="dataset_download_ibtracs", 
    #     provenance_save_dir="prov",
    #     disable_codecarbon=True
    # )

    # ibtracs_path = os.path.join(DATA_PATH, "ibtracs.since1980.list.v04r01.csv")
    # ibtracs = pd.read_csv(ibtracs_path, low_memory=False)
    # ibtracs = ibtracs[ibtracs["Year"].isin([2020, 2021])]

    # bounds = gpd.read_file("data/CNTR_BN_60M_2024_3035.geojson")
    # bounds = bounds[bounds["EU_FLAG"] == "T"]

    era5_path = os.path.join(DATA_PATH, "data.grib")
    era5 = xr.open_dataset(era5_path, engine="cfgrib") #(8768, 121, 161)
    fig, ax = plt.subplots(figsize=(10, 10))
    era5.v10[0].plot(ax=ax)
    plt.show()

    # import os
    # import xarray as xr
    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation, PillowWriter

    # era5_path = os.path.join(DATA_PATH, "data.grib")
    # era5 = xr.open_dataset(era5_path, engine="cfgrib")

    # data = era5.v10  # (time, lat, lon)
    # print(data.shape)
    # data = data.isel(time=slice(0, 100, 1))

    # fig, ax = plt.subplots(figsize=(16, 10))

    # # Initial frame
    # img = data.isel(time=0).plot(ax=ax, add_colorbar=True)
    # ax.set_title(f"MSL – timestep 0")

    # def update(frame):
    #     ax.clear()
    #     data.isel(time=frame).plot(ax=ax, add_colorbar=False)
    #     ax.set_title(f"MSL – timestep {frame}")
    #     return ax

    # ani = FuncAnimation(
    #     fig,
    #     update,
    #     frames=len(data.time),
    #     interval=50,   # ms between frames (increase if too fast)
    # )

    # # Save as GIF
    # ani.save(
    #     "msl_animation.gif",
    #     writer=PillowWriter(fps=10)
    # )

    # plt.close()



if __name__ == "__main__": 
    main()