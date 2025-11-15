import os
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp  


class BalloonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.masks = []
        self.transform = transform

        with open(os.path.join(root_dir, "via_region_data.json"), "r") as f:
            data = json.load(f)

        for key in data:
            file_path = os.path.join(root_dir, data[key]["filename"])
            mask = np.zeros(cv2.imread(file_path).shape[:2], dtype=np.uint8)
            for region in data[key]["regions"]:
                points_x = data[key]["regions"][region]["shape_attributes"]["all_points_x"]
                points_y = data[key]["regions"][region]["shape_attributes"]["all_points_y"]
                pts = np.array(list(zip(points_x, points_y)), np.int32)
                cv2.fillPoly(mask, [pts], 1)
            self.images.append(file_path)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        img = torch.tensor(img, dtype=torch.float).permute(2,0,1)/255.0
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
        return img, mask


class BallonSegmentator_Unet:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):
        import segmentation_models_pytorch as smp
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        self.model.to(self.device)

    def train(self, train_dataset, val_dataset=None, epochs=10, batch_size=8, lr=1e-3, save_path="unet_balloon.pth"):
        if self.model is None:
            self.build_model()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

            if val_dataset:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for imgs, masks in val_loader:
                        imgs, masks = imgs.to(self.device), masks.to(self.device)
                        outputs = self.model(imgs)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {avg_val_loss:.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved: {save_path}")

    def load_model(self, model_path):
        
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  
            in_channels=3,
            classes=1
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model {model_path} başarıyla yüklendi ve eval moduna geçirildi.")

    def predict_area_percent(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_img = cv2.resize(img_rgb, (256, 256))
        input_tensor = torch.tensor(input_img, dtype=torch.float).permute(2,0,1).unsqueeze(0)/255.0
        input_tensor = input_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

        area_percent = pred_mask_bin.sum() / (pred_mask_bin.shape[0]*pred_mask_bin.shape[1]) * 100

        pred_mask_full = cv2.resize(pred_mask_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = img_rgb.copy()
        overlay[pred_mask_full==1] = [0,255,0]  # yeşil
        alpha = 0.5
        output_img = cv2.addWeighted(overlay, alpha, img_rgb, 1-alpha, 0)

        plt.figure(figsize=(8,8))
        plt.imshow(output_img)
        plt.title(f"Predicted Balloon Area: {area_percent:.2f}%")
        plt.axis('off')
        plt.show()

        return area_percent
    
    def evaluate_ap50(self, dataset, threshold=0.5):
       
        self.model.eval()
        ious = []
        tps = 0
        fps = 0

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for img, gt_mask in loader:
                img = img.to(self.device)
                gt_mask = gt_mask.squeeze().cpu().numpy()

                pred = self.model(img)
                pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
                pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

                intersection = np.logical_and(pred_mask_bin, gt_mask).sum()
                union = np.logical_or(pred_mask_bin, gt_mask).sum()

                if union == 0:
                    continue  

                iou = intersection / union
                ious.append(iou)

                if iou >= threshold:
                    tps += 1
                else:
                    fps += 1

        total_gt = len(dataset)
        recall = tps / total_gt
        precision = tps / (tps + fps + 1e-6)

        ap50 = precision * recall  

        print(f"Total GT: {total_gt}")
        print(f"TP: {tps}, FP: {fps}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"AP@0.5: {ap50:.4f}")

        return ap50
