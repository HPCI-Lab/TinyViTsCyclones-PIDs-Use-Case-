
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch 
import yprov4ml
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse
import yaml
import matplotlib.pyplot as plt

from model.reconstruction_head import ClassificationHead 
from consts import WEIGHTS_PATH, DEVICE, DATA_PATH, IMGS_PATH
from climate_datasets.era5_cyclone import Era5CycloneDataset

EPOCHS = 1

def main(yaml_file_path): 

    configs = yaml.safe_load(open(yaml_file_path, mode="rb"))

    yprov4ml.start_run(
        experiment_name=f"model_finetune_{configs["model_size"]}", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    # model = tiny_vit.tiny_vit_custom(pretrained=False)
    model_path = os.path.join(WEIGHTS_PATH, f"tiny_vit_{configs["model_size"]}_pretrained.pt")
    model = torch.load(model_path, weights_only=False)
    if configs["model_size"] == "small": 
        model.head = ClassificationHead(320, 1)
    elif configs["model_size"] == "medium": 
        model.head = ClassificationHead(448, 1)
    elif configs["model_size"] == "medium": 
        model.head = ClassificationHead(576, 1)
    
    model = model.to(DEVICE)
    yprov4ml.log_model(f"tiny_vit_{configs["model_size"]}_pretrained", model, is_input=True, context=None)

    era5_path = os.path.join(DATA_PATH, "data.grib")
    yprov4ml.log_artifact("era5_path", era5_path, is_input=True)
    ibtracs_path = os.path.join(DATA_PATH, "ibtracs.since1980.list.v04r01.csv")
    yprov4ml.log_artifact("ibtracs_dataset", ibtracs_path, is_input=True)

    train_dataset = Era5CycloneDataset(split="train", patch_size=96)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = CrossEntropyLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(EPOCHS): 
        for X, y in tqdm(train_dataloader): 
            X = X.unsqueeze(1).to(DEVICE)
            y = y.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            y_hat, _ = model(X)
            loss = criterion(y_hat, y)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

    # plt.plot(losses)
    # pth = os.path.join(IMGS_PATH, f"losses_{configs["model_size"]}_finetune.png")
    # plt.savefig(pth)
    # plt.close()
    # yprov4ml.log_artifact(f"losses_{configs["model_size"]}_finetune", pth, is_input=False)

    model_path = os.path.join(WEIGHTS_PATH, f"tiny_vit_{configs["model_size"]}_finetuned.pt")
    torch.save(model, model_path)
    yprov4ml.log_model(f"tiny_vit_{configs["model_size"]}_finetuned", model, is_input=False, context=None)

    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml')
    args = parser.parse_args()
    main(args.yaml)