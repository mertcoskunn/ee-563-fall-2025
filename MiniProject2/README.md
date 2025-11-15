# Balloon Detection Project

This repository contains two object detection approaches (YOLOv8 and U-Net) trained on the Balloon dataset to estimate the balloon area percentage in images.

---

## ⚠️ Dataset Preparation
YOLO and U-Net use **different dataset directory structures**.  
For this reason, **duplicate the dataset folder** and rename the copy as: balloon_1/

One version will be used for YOLO, and the other for U-Net.

## 📥 Installation (Required Packages)
```bash
pip install ultralytics
pip install segmentation-models-pytorch
pip install albumentations
```
---

## 🚀 Training

### **Train U-Net**
```bash
python train_unet.py
```

### **Train YOLO**
```bash
python train_yolo.py
```

## 🔍 Prediction

### **Predict with U-Net**
```bash
python predict_unet.py IMAGE_PATH
```

### **Predict with YOLO**
```bash
python predict_yolo.py IMAGE_PATH
```

## 📊 Model Evaluation
```bash
python evaluationModels.py
```

## ⏱️ Inference Time Measurement

### **U-Net Inference Time**
```bash
python inference_unet.py
```

### **YOLO Inference Time**
```bash
python inference_yolo.py
```
