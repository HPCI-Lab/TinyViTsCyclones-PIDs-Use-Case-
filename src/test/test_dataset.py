
import matplotlib.pyplot as plt
import sys
sys.path.append("./src")

from torch.utils.data import DataLoader
from climate_datasets.era5 import Era5Dataset

def main(): 
    train_dataset = Era5Dataset(split="train", patch_size=96)
    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    for i, (X, y) in enumerate(train_dataloader): 
        print(i, X[0].shape)
        if i > 10: break
        plt.imshow(X[0])
        plt.savefig(f"{i}.png")
        plt.close()


if __name__ == "__main__": 
    main()