from multiprocessing import freeze_support
from model import get_model
from data import get_data_loaders
from utils import load_checkpoint, evaluate_on_device 
import os
import torch



def main():

    data_Dir = "hymenoptera_data"
    parent_path = os.path.abspath("..")
    data_path = os.path.join(parent_path, data_Dir)
    val_path = os.path.join(data_path, "val")
    train_path = os.path.join(data_path, "train")
    
    train_loader, val_loader = get_data_loaders(
            train_path,
            val_path,
            batch_size=32
        )

    parent_path = os.path.abspath("..")

    model_path =  data_path = os.path.join(parent_path, "my_models")
    prefix = "resnet18_cuda_lr0.1_mom0.5"
    model_path = os.path.join(model_path, f"{prefix}_epoch8.pth")
        
    model = get_model("resnet18", num_classes=2, pretrained=False)

    device = "cuda"
    epoch_loaded = load_checkpoint(model_path, model, device=device)

    evaluate_on_device(model, val_loader, torch.device(device), repeat=20)

    model = get_model("resnet18", num_classes=2, pretrained=False)

    device = "cpu"
    epoch_loaded = load_checkpoint(model_path, model, device=device)

    evaluate_on_device(model, val_loader, torch.device(device), repeat=20)


if __name__ == "__main__":
    freeze_support()  
    main()