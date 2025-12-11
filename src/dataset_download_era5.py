

import os
import requests
from tqdm import tqdm
import yprov4ml

from consts import DATA_PATH

def main(): 
    yprov4ml.start_run(
        experiment_name="dataset_download_era5", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    start_year = 2020
    end_year = 2022
    era5_url = ""
    era5_path = os.path.join(DATA_PATH, "era5")

    yprov4ml.log_param("start_year", start_year)
    yprov4ml.log_param("end_year", end_year)


    yprov4ml.log_artifact("era5_path", era5_path, is_input=False)

    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    main()