
import torch 
import os
import yprov4ml
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import argparse
import yaml
from tqdm import tqdm

from consts import WEIGHTS_PATH, DEVICE, DATA_PATH
from climate_datasets.era5_cyclone import Era5CycloneDataset

def main(yaml_file_path): 

    configs = yaml.safe_load(open(yaml_file_path, mode="rb"))

    yprov4ml.start_run(
        experiment_name=f"evaluate_finetune_{configs["model_size"]}", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    era5_path = os.path.join(DATA_PATH, "data.grib")
    yprov4ml.log_artifact("era5_path", era5_path, is_input=True)
    ibtracs_path = os.path.join(DATA_PATH, "ibtracs.since1980.list.v04r01.csv")
    yprov4ml.log_artifact("ibtracs_dataset", ibtracs_path, is_input=True)

    test_dataset = Era5CycloneDataset(split="test", patch_size=96)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model_path = os.path.join(WEIGHTS_PATH, f"tiny_vit_{configs["model_size"]}_finetuned.pt")
    model = torch.load(model_path, weights_only=False).to(DEVICE)
    yprov4ml.log_model(f"tiny_vit_{configs["model_size"]}_finetuned", model, is_input=True, context=None)

    criterion = CrossEntropyLoss().to(DEVICE)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            X = X.unsqueeze(1).to(DEVICE)
            y = y.unsqueeze(1).to(DEVICE)

            logits, _ = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    yprov4ml.log_param("total_loss", total_loss)
    yprov4ml.log_param("avg_loss", avg_loss)
    yprov4ml.log_param("total_samples", total)
    yprov4ml.log_param("accuracy", accuracy)
    yprov4ml.log_param("correct", correct)

    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml')
    args = parser.parse_args()
    main(args.yaml)