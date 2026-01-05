

import os
import requests
from tqdm import tqdm
import yprov4ml

from consts import DATA_PATH

def main(): 
    yprov4ml.start_run(
        experiment_name="dataset_download_ibtracs", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    ibtracs_url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
    ibtracs_path = os.path.join(DATA_PATH, "ibtracs.since1980.list.v04r01.csv")
    yprov4ml.log_param("ibtracs_url", ibtracs_url)

    if not os.path.exists(ibtracs_path): 
        response = requests.get(ibtracs_url)
        file = open(ibtracs_path, "wb")
        for data in tqdm(response.iter_content(chunk_size=1024), unit="kB"):
            file.write(data)

    yprov4ml.log_artifact("ibtracs_dataset", ibtracs_path, is_input=False)

    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    main()