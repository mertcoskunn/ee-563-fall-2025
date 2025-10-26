import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import load_checkpoint 
from data import get_data_loaders
from model import get_model
from PIL import Image
import random
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    
    save_dir = "hymenoptera_data"
    parent_path = os.path.abspath("..")
    data_path = os.path.join(parent_path, save_dir)
    dataset = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    
    # --- If you want to use a different image, change it here ---
    # To select a random image from the dataset, use this line:
    # image_path, label = random.choice(dataset.imgs)

    # To select a specific image manually:
    # image_path = "path/to/your/image.jpg"
    # label = 0  # manually specify the class label (e.g., 0 or 1)
    image_path, label = random.choice(dataset.imgs)  
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)  

    model_path =  data_path = os.path.join(parent_path, "my_models")
    prefix = "resnet18_cuda_lr0.1_mom0.5"
    model_path = os.path.join(model_path, f"{prefix}_epoch8.pth")
    
    model = get_model("resnet18", num_classes=2, pretrained=False)
    epoch_loaded = load_checkpoint(model_path, model, device=device)
    model.to(device)
    model.eval()

    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

    print(f"Actual label: {idx_to_class[label]}, Predicted label: {idx_to_class[predicted.item()]}")
   
if __name__ == "__main__":
    main()