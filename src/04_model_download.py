
import torch 
import os
import yprov4ml

from model import tiny_vit
from consts import WEIGHTS_PATH

def main(): 
    yprov4ml.start_run(
        experiment_name="pretrained_download", 
        provenance_save_dir="prov",
        disable_codecarbon=True
    )

    model = tiny_vit.tiny_vit_5m_224(pretrained=True)
    model_path = os.path.join(WEIGHTS_PATH, "tiny_vit_5m_224.pt")
    torch.save(model, model_path)
    yprov4ml.log_model("tiny_vit_5m_224", model)
    
    yprov4ml.end_run(True, True, False)

if __name__ == "__main__": 
    main()