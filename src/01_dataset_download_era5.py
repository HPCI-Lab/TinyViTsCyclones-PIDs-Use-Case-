

import os
import requests
from tqdm import tqdm
import yprov4ml

from consts import DATA_PATH

def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

def main(): 
    yprov4ml.start_run(
        experiment_name="dataset_download_era5", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    start_year = 2020
    end_year = 2022
    era5_url = "https://drive.usercontent.google.com/download?id=1uDxTZixWV1EHlSMYFSk7SoZ52f0MuQAE&export=download&authuser=1"
    era5_path = os.path.join(DATA_PATH, "era5")

    era5_path = download_file(era5_url, era5_path)

    yprov4ml.log_param("start_year", start_year)
    yprov4ml.log_param("end_year", end_year)

    yprov4ml.log_artifact("era5_path", era5_path, is_input=False)

    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    main()