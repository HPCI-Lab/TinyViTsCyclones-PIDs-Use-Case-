import torch.nn as nn
import torch.nn.functional as F

class ReconstructionHead(nn.Module):
    def __init__(self, in_features=320, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.fc = nn.Linear(in_features, patch_size * patch_size)

    def forward(self, x):
        # x: [B, 320]
        x = self.fc(x)                 # [B, 1024]
        x = x.view(x.size(0), 1, self.patch_size, self.patch_size)
        return x