
import torch 
import os
import yprov4ml
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from model import tiny_vit
from consts import WEIGHTS_PATH, DEVICE
from climate_datasets.era5_cyclone import Era5CycloneDataset

def main(): 
    yprov4ml.start_run(
        experiment_name="evaluate_finetuned", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    test_dataset = Era5CycloneDataset(split="test")
    test_dataloader = DataLoader(test_dataset)

    model = tiny_vit.tiny_vit_5m_224(pretrained=False)
    model_path = os.path.join(WEIGHTS_PATH, "tiny_vit_5m_224_finetuned.pt")
    torch.load(model, model_path).to(DEVICE)

    criterion = CrossEntropyLoss().to(DEVICE)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
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