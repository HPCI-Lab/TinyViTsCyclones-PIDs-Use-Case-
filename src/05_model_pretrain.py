
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch 
import yprov4ml
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import yaml

from model import tiny_vit
from consts import WEIGHTS_PATH, DEVICE, IMGS_PATH
from climate_datasets.era5 import Era5Dataset
from model.reconstruction_head import ReconstructionHead

EPOCHS = 1

def main(yaml_file_path): 

    configs = yaml.safe_load(open(yaml_file_path, mode="rb"))

    yprov4ml.start_run(
        experiment_name=f"model_pretrain_{configs["model_size"]}", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    if configs["model_size"] == "small": 
        model = tiny_vit.tiny_vit_c_small(pretrained=False).to(DEVICE)
    elif configs["model_size"] == "medium": 
        model = tiny_vit.tiny_vit_c_medium(pretrained=False).to(DEVICE)
    elif configs["model_size"] == "large": 
        model = tiny_vit.tiny_vit_c_large(pretrained=False).to(DEVICE)
    model.train()

    train_dataset = Era5Dataset(split="train", patch_size=96)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = MSELoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.0001)

    losses = []
    for _ in range(EPOCHS): 
        for X, y in tqdm(train_dataloader): 
            optimizer.zero_grad()

            X = X.unsqueeze(1).to(DEVICE)
            y = y.unsqueeze(1).to(DEVICE)

            y_hat, _ = model(X)
            loss = criterion(y_hat, X)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

    plt.plot(losses)
    pth = os.path.join(IMGS_PATH, f"losses_{configs["model_size"]}.png")
    plt.savefig(pth)
    plt.close()
    yprov4ml.log_artifact(f"losses_{configs["model_size"]}", pth)

    model_path = os.path.join(WEIGHTS_PATH, f"tiny_vit_{configs["model_size"]}_pretrained.pt")
    torch.save(model, model_path)
    yprov4ml.log_model(f"tiny_vit_{configs["model_size"]}_pretrained", model)

    yprov4ml.end_run(True, True, False)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml')
    args = parser.parse_args()
    main(args.yaml)