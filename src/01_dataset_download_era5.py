

import os
import requests
from tqdm import tqdm
import yprov4ml

from consts import DATA_PATH

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None

def main(): 
    yprov4ml.start_run(
        experiment_name="dataset_download_era5", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    start_year = 2020
    end_year = 2022
    era5_url = "https://drive.usercontent.google.com/download?id=1uDxTZixWV1EHlSMYFSk7SoZ52f0MuQAE&export=download&confirm=1"
    era5_path = os.path.join(DATA_PATH, "data.grib")
    file_id = "1uDxTZixWV1EHlSMYFSk7SoZ52f0MuQAE"

    session = requests.Session()

    response = session.get(era5_url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(era5_url, params=params, stream=True)

    save_response_content(response, era5_path)

    yprov4ml.log_param("start_year", start_year)
    yprov4ml.log_param("end_year", end_year)

    yprov4ml.log_artifact("era5_path", era5_path, is_input=False)

    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    main()