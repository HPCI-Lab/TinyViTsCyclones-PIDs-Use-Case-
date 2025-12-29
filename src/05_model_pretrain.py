
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

from model import tiny_vit
from consts import WEIGHTS_PATH, DEVICE
from climate_datasets.era5 import Era5Dataset

EPOCHS = 1

def main(): 

    yprov4ml.start_run(
        experiment_name="pretrained_finetune", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

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
    plt.savefig("losses.png")
    plt.close()

    # model.eval()
    # for i, (X, y) in enumerate(tqdm(train_dataloader)): 
    #     if i > 10: break
    #     X = X.unsqueeze(1).to(DEVICE)
    #     y = y.unsqueeze(1).to(DEVICE)

    #     y_hat, _ = model(X)
    #     y_hat = (y_hat.detach().cpu().numpy().squeeze() * train_dataloader.dataset.MAX) + train_dataloader.dataset.MIN

    #     plt.imshow(X[0].detach().cpu().numpy().squeeze())
    #     plt.savefig(f"in_{i}.png")
    #     plt.close()
    #     plt.imshow(y_hat[0])
    #     plt.savefig(f"out_{i}.png")
    #     plt.close()

    model_path = os.path.join(WEIGHTS_PATH, "tiny_vit_5m_224_pretrained.pt")
    torch.save(model, model_path)
    yprov4ml.log_model("tiny_vit_5m_224_pretrained", model)

    yprov4ml.end_run(True, True, False)


if __name__ == "__main__": 
    main()