
import matplotlib.pyplot as plt
import sys
sys.path.append("./TinyViTsCyclones-PIDs-Use-Case-/src")

from torch.utils.data import DataLoader

from climate_datasets.era5 import Era5Dataset
from climate_datasets.era5_cyclone import Era5CycloneDataset

def main(): 
    train_dataset = Era5CycloneDataset(split="train", patch_size=96)
    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for i, (X, y) in enumerate(train_dataloader): 
        # if i > 10: break
        print(i, y)
        # plt.imshow(X[0])
        # plt.savefig(f"i{i}.png")
        # plt.close()

if __name__ == "__main__": 
    main()