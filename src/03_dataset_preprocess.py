
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import yprov4ml
import geopandas as gpd

from consts import DATA_PATH

latitude = 161
longitude = 81

def main(): 
    yprov4ml.start_run(
        experiment_name="dataset_preprocess", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    ibtracs_path = os.path.join(DATA_PATH, "ibtracs.since1980.list.v04r01.csv")
    ibtracs = pd.read_csv(ibtracs_path, low_memory=False)
    ibtracs = ibtracs[ibtracs["SEASON"].isin([2020, 2021])]

    era5_path = os.path.join(DATA_PATH, "data.grib")
    # if not working then install via conda: conda install -c conda-forge cfgrib
    era5 = xr.open_dataset(era5_path, engine="cfgrib") #(8768, 121, 161)
    print(era5)


if __name__ == "__main__": 
    main()