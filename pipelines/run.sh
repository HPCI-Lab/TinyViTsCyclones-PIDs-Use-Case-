
pip install -r requirements.txt

python src/01_dataset_download_era5.py

python src/02_dataset_download_ibtracs.py

python src/03_dataset_preprocess.py

python src/04_model_download.py

python src/05_model_pretrain.py

python src/06_model_finetune.py

python src/07_evaluate_pretrained.py

python src/08_evaluate_finetuned.py