
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch 
import torch.nn as nn
import os
import yprov4ml
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from model import tiny_vit
from consts import WEIGHTS_PATH, DEVICE
from climate_datasets.era5_cyclone import Era5CycloneDataset

EPOCHS = 10

def main(): 
    yprov4ml.start_run(
        experiment_name="pretrained_finetune", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    # model = tiny_vit.tiny_vit_custom(pretrained=False)
    model_path = os.path.join(WEIGHTS_PATH, "tiny_vit_5m_224_pretrained.pt")
    model = torch.load(model_path, weights_only=False)
    model.head = nn.Linear(576, 1)
    model = model.to(DEVICE)

    train_dataset = Era5CycloneDataset(split="train", patch_size=96)
    # train_dataset = Subset(train_dataset, range(64*75))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = CrossEntropyLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.01)

    for _ in range(EPOCHS): 
        for X, y in tqdm(train_dataloader): 
            X = X.unsqueeze(1).to(DEVICE)
            y = y.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            y_hat, _ = model(X)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()

    model_path = os.path.join(WEIGHTS_PATH, "tiny_vit_5m_224_finetuned.pt")
    torch.save(model, model_path)

    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    main()