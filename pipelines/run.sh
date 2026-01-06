#!/usr/bin/env bash

check_step () {
    if [ $? -ne 0 ]; then
        echo "❌ Error during: $1"
        exit 1
    fi
}

# pip install -r requirements.txt
# check_step "pip install"
# echo "✅ pip install completed successfully!"

# conda install -c conda-forge cfgrib
# check_step "cfgrib install"
# echo "✅ cfgrib install completed successfully!"

python src/01_dataset_download_era5.py
check_step "ERA5 download"
echo "✅ ERA5 download completed successfully!"

python src/02_dataset_download_ibtracs.py
check_step "IBTrACS download"
echo "✅ IBTrACS download completed successfully!"

# python src/03_dataset_preprocess.py
# check_step "Dataset preprocessing"
# echo "✅ Dataset preprocessing completed successfully!"

# python src/04_model_download.py
# check_step "Model download"
# echo "✅ Model download completed successfully!"

python src/05_model_pretrain.py pipelines/small.yaml
check_step "Model pretraining"
echo "✅ Model pretraining completed successfully!"

python src/06_model_finetune.py pipelines/small.yaml
check_step "Model finetuning"
echo "✅ Model finetuning completed successfully!"

python src/07_evaluate_pretrained.py pipelines/small.yaml
check_step "Pretrained evaluation"
echo "✅ Pretrained evaluation completed successfully!"

python src/08_evaluate_finetuned.py pipelines/small.yaml
check_step "Finetuned evaluation"
echo "✅ Finetuned evaluation completed successfully!"

echo "✅ All steps completed successfully!"