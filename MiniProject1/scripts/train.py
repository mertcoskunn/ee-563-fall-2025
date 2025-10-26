import torch
from utils import save_checkpoint
from tqdm import tqdm 



def train_model(model, train_loader, val_loader, device, optimizer, criterion, epochs=10, 
                save_dir=None, save_every=1, prefix="model", freeze_backbone=False):
    """
    model           : torch.nn.Module
    train_loader    : training DataLoader
    val_loader      : validation DataLoader
    device          : "cuda" veya "cpu"
    optimizer       : optimizer instance (SGD, Adam)
    criterion       : loss function (nn.CrossEntropyLoss vb.)
    epochs          : kaç epoch çalıştırılacağı
    freeze_backbone : True olursa sadece fc eğitilir
    save_dir        : checkpoint ve history kaydı için klasör
    save_every      : kaç epoch'ta bir kaydedilecek
    prefix          : kaydedilen dosya isimleri için prefix
    """
    # --- Freeze backbone if needed ---
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        # optimizer'ı sadece fc için al
        optimizer = type(optimizer)(model.fc.parameters(), lr=getattr(optimizer, 'defaults', {}).get('lr', 1e-3))
        print("Backbone frozen. Only fc layer will be trained.")

    model.to(device)

    # History kaydı
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # --- Training ---
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                print("probs top1:", probs.max(1)[0][:10])
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        print(f"Epoch {epoch}/{epochs}: "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        # --- Save checkpoint ---
        if save_dir is not None and (epoch % save_every == 0 or epoch == epochs):
            save_checkpoint(save_dir, model, optimizer, epoch, history, prefix=prefix)

    return model, history


### example loop for training
from model import get_model
from data import get_data_loaders
from torchvision import datasets
from losses import FocalLoss
import torch.nn as nn
import torch.optim as optim
import os

if __name__ == "__main__":
    save_dir = "hymenoptera_data"
    parent_path = os.path.abspath("..")
    data_path = os.path.join(parent_path, save_dir)

    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")

    train_loader, val_loader = get_data_loaders(
        train_path,
        val_path, 
        batch_size=32
    )

    # model parameters
    # You can change device, learning rates, momentums, backbones, and data augmentation modes here
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [0.05, 0.01, 0.1]
    momentums = [0.5, 0.9, 0.99]
    back_bones = ["resnet18","resnet34", "resnet50"]
    data_augments_modes = ["none", "light", "strong"]
    epochs = 8

    save_dir = "my_models"
    parent_path = os.path.abspath("..")
    model_path = os.path.join(parent_path, save_dir)
    os.makedirs(model_path, exist_ok=True)

    for lr in learning_rates:
        for mom in momentums:
                for b in back_bones:
                    for d in data_augments_modes:
                        prefix = f"{b}_{d}_{device}_lr{lr}_mom{mom}"  
                        print(f"\n=== Training: {prefix} ===")

                        model = get_model(b, num_classes=2, pretrained=True)
                        model.to(device)

                        # You can switch between loss functions here:
                        criterion = nn.CrossEntropyLoss()
                        #criterion = FocalLoss(alpha=1.0, gamma=2.0)

                        # You can switch between optimizers here:
                        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
                        #optimizer = optim.Adam(model.parameters(), lr=lr)
                        model, history = train_model(
                            model, train_loader, val_loader, device,
                            optimizer, criterion, epochs=epochs,
                            save_dir=save_dir, save_every=1, prefix=prefix, freeze_backbone = True
                        )