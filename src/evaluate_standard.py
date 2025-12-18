
import torch 
import os
import yprov4ml
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from model import tiny_vit
from consts import WEIGHTS_PATH
from climate_datasets.era5 import Era5Dataset

def main(): 
    yprov4ml.start_run(
        experiment_name="evaluate_standard", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    test_dataset = Era5Dataset(split="test")
    test_dataloader = DataLoader(test_dataset)

    model = tiny_vit.tiny_vit_5m_224(pretrained=False)
    model_path = os.path.join(WEIGHTS_PATH, "tiny_vit_5m_224_finetuned.pt")
    torch.load(model, model_path)

    criterion = CrossEntropyLoss()

    model.eval()
    for X, y in test_dataloader: 
        y_hat = model(X)
        loss = criterion(y_hat, y)

    yprov4ml.end_run(True, True, False)