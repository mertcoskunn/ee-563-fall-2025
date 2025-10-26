# plot_lr_vs_mom.py
import os
import json
import matplotlib.pyplot as plt


# --- Note ---
# To visualize a specific plot, uncomment the corresponding block below 
# (Validation Loss/Accuracy for different Data Augmentation modes or Optimizers) 
# and run the code.


save_dir = "my_models"
parent_path = os.path.abspath("..")
data_path = os.path.join(parent_path, save_dir)

# ---------- Start ----------

# device = "cuda"
# learning_rates = [0.05, 0.01, 0.1]
# momentums = [0.5, 0.9, 0.99]

# ---------- Validation Loss ----------
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 satır, 2 sütun

# # ---------- Validation Loss ----------
# axes[0].set_title("Validation Loss (ResNet-18)")
# for lr in learning_rates:
#     for mom in momentums:
#         prefix = f"resnet18_{device}_lr{lr}_mom{mom}"
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
        
#         if os.path.exists(history_path):
#             with open(history_path, "r") as f:
#                 history = json.load(f)
#             axes[0].plot(history["val_loss"], label=f"lr={lr}, mom={mom}")
#         else:
#             print(f"File not found: {history_path}")

# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("Validation Loss")
# axes[0].grid(True)
# axes[0].legend()

# # ---------- Validation Acc ----------
# axes[1].set_title("Validation Accuracy (ResNet-18)")
# for lr in learning_rates:
#     for mom in momentums:
#         prefix = f"resnet18_{device}_lr{lr}_mom{mom}"
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
        
#         if os.path.exists(history_path):
#             with open(history_path, "r") as f:
#                 history = json.load(f)
#             axes[1].plot(history["val_acc"], label=f"lr={lr}, mom={mom}")
#         else:
#             print(f"File not found: {history_path}")

# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Validation Accuracy")
# axes[1].grid(True)
# axes[1].legend()

# plt.suptitle("Validation Metrics for different momentums and learning rates (ResNet-18)", fontsize=14)
# plt.tight_layout()
# plt.show()
# ---------- End ----------

# # ---------- Start ----------

# device = "cuda"
# learning_rates = [0.05, 0.01, 0.1]
# momentum_to_plot = 0.9

# # ---------- Validation Loss ----------
# plt.figure(figsize=(10,5))
# plt.title(f"Validation Loss for different Learning Rates (momentum={momentum_to_plot}, backbone resnet-18)")

# for lr in learning_rates:
#     prefix = f"resnet18_{device}_lr{lr}_mom{momentum_to_plot}"
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    
#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         plt.plot(history["val_loss"], label=f"LR={lr}")
#     else:
#         print(f"File not found: {history_path}")

# plt.xlabel("Epoch")
# plt.ylabel("Validation Loss")
# plt.grid(True)
# plt.legend()
# plt.show()


# # ---------- Validation Acc ----------
# plt.figure(figsize=(10,5))
# plt.title(f"Validation Acc for different Learning Rates (momentum={momentum_to_plot}, backbone resnet-18)")

# for lr in learning_rates:
#     prefix = f"resnet18_{device}_lr{lr}_mom{momentum_to_plot}"
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    
#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         plt.plot(history["val_acc"], label=f"LR={lr}")
#     else:
#         print(f"File not found: {history_path}")

# plt.xlabel("Epoch")
# plt.ylabel("Validation Acc")
# plt.grid(True)
# plt.legend()
# plt.show()

# # ---------- End ----------



# # ---------- Start ----------
# device = "cuda"
# learning_rate = 0.1
# momentum_to_plot = 0.5
# back_bones = ["resnet18", "resnet34", "resnet50"]

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))  

# # ---------- Validation Loss ----------
# axes[0].set_title(f"Validation Loss (momentum={momentum_to_plot}, lr={learning_rate})")

# for b in back_bones:
#     prefix = f"{b}_{device}_lr{learning_rate}_mom{momentum_to_plot}"
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")

#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         axes[0].plot(history["val_loss"], label=f"{b}")
#     else:
#         prefix = f"{b}_light_{device}_lr{learning_rate}_mom{momentum_to_plot}"
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")

#         if os.path.exists(history_path):
#             with open(history_path, "r") as f:
#                 history = json.load(f)
#             axes[0].plot(history["val_loss"], label=f"{b}")
#         else:
#             print("File not found")


# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("Validation Loss")
# axes[0].grid(True)
# axes[0].legend()

# # ---------- Validation Accuracy ----------
# axes[1].set_title(f"Validation Accuracy (momentum={momentum_to_plot}, lr={learning_rate})")

# for b in back_bones:
#     prefix = f"{b}_{device}_lr{learning_rate}_mom{momentum_to_plot}"
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")

#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         axes[1].plot(history["val_acc"], label=f"{b}")
#     else:
#         prefix = f"{b}_light_{device}_lr{learning_rate}_mom{momentum_to_plot}"
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")

#         if os.path.exists(history_path):
#             with open(history_path, "r") as f:
#                 history = json.load(f)
#             axes[1].plot(history["val_acc"], label=f"{b}")
#         else:
#             print("File not found")

# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Validation Accuracy")
# axes[1].grid(True)
# axes[1].legend()

# plt.suptitle(f"Validation Metrics for Different Backbones (momentum={momentum_to_plot}, lr={learning_rate})", fontsize=14)
# plt.tight_layout()
# plt.show()
# # ---------- End ----------


# # ---------- Start ----------
# device = "cuda"
# learning_rate = 0.1
# momentum_to_plot = 0.5
# back_bone = "resnet18"
# data_augments_modes = ["none", "light" , "strong"]
# line_styles = [":", "--", "-."]
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 satır, 2 sütun

# # ---------- Validation Loss ----------
# axes[0].set_title(f"Validation Loss (backbone={back_bone}, mom={momentum_to_plot}, lr={learning_rate})")

# for d, style  in  zip(data_augments_modes, line_styles):
#     prefix = f"{back_bone}_{d}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    
#     if not os.path.exists(history_path):
#         # Eğer o augment mode yoksa fallback olarak _d_ olmayan dosyayı dene
#         prefix = f"{back_bone}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")

#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         axes[0].plot(history["val_loss"], label=f"DA={d}", linestyle=style)
#     else:
#         print(f"File not found: {history_path}")

# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("Validation Loss")
# axes[0].grid(True)
# axes[0].legend()

# # ---------- Validation Accuracy ----------
# axes[1].set_title(f"Validation Accuracy (backbone={back_bone}, mom={momentum_to_plot}, lr={learning_rate})")

# for d, style  in  zip(data_augments_modes, line_styles):
#     prefix = f"{back_bone}_{d}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    
#     if not os.path.exists(history_path):
#         prefix = f"{back_bone}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")

#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         axes[1].plot(history["val_acc"], label=f"DA={d}",linestyle=style)
#     else:
#         print(f"File not found: {history_path}")

# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Validation Accuracy")
# axes[1].grid(True)
# axes[1].legend()

# plt.suptitle(f"Validation Metrics for Different Data Augmentation Modes\n(backbone={back_bone}, mom={momentum_to_plot}, lr={learning_rate})", fontsize=14)
# plt.tight_layout()
# plt.show()
# # ---------- End ----------


# ---------- Start ----------
# device = "cuda"
# learning_rate = 0.1
# momentum_to_plot = 0.5
# back_bone = "resnet18"
# data_augments_mode = "light"
# optimizers = ["sgd", "adam"]

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 satır, 2 sütun
# # ---------- Validation Loss ----------
# axes[0].set_title(f"Validation Loss (backbone={back_bone}, LR={learning_rate}, momentum={momentum_to_plot})")

# for opt in optimizers:
#     prefix = f"{back_bone}_{data_augments_mode}_{opt}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    
#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         axes[0].plot(history["val_loss"], label=f"Optimizer={opt}")
#     else:
#         prefix = f"{back_bone}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
        
#         if os.path.exists(history_path):
#             with open(history_path, "r") as f:
#                 history = json.load(f)
#             axes[0].plot(history["val_loss"], label=f"Optimizer={opt}")

# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("Validation Loss")
# axes[0].grid(True)
# axes[0].legend()

# # ---------- Validation Accuracy ----------
# axes[1].set_title(f"Validation Accuracy (backbone={back_bone}, LR={learning_rate}, momentum={momentum_to_plot})")

# for opt in optimizers:
#     prefix = f"{back_bone}_{data_augments_mode}_{opt}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#     history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    
#     if os.path.exists(history_path):
#         with open(history_path, "r") as f:
#             history = json.load(f)
#         axes[1].plot(history["val_acc"], label=f"Optimizer={opt}")
#     else:
#         prefix = f"{back_bone}_{device}_lr{learning_rate}_mom{momentum_to_plot}"  
#         history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
        
#         if os.path.exists(history_path):
#             with open(history_path, "r") as f:
#                 history = json.load(f)
#             axes[1].plot(history["val_acc"], label=f"Optimizer={opt}")

# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Validation Accuracy")
# axes[1].grid(True)
# axes[1].legend()

# plt.suptitle(f"Validation Metrics for Different Optimizers\n(backbone={back_bone}, LR={learning_rate}, momentum={momentum_to_plot})", fontsize=14)
# plt.tight_layout()
# plt.show()
# ---------- End ----------



# ---------- Start ----------
device = "cuda"
learning_rate = 0.01
momentum_to_plot = 0.9
back_bone = "resnet18"
data_augments_mode= "light"
loss_functions = ["FocalLoss", "CrossEntropy"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 satır, 2 sütun

# ---------- Validation Loss ----------
axes[0].set_title(f"Validation Loss (backbone={back_bone}, momentum={momentum_to_plot}, DA={data_augments_mode})")

for loss in loss_functions:
    if loss == "FocalLoss":
        prefix = f"{back_bone}_{data_augments_mode}_FocalLoss_{device}_lr{learning_rate}_mom{momentum_to_plot}"
        label = "Focal Loss"
    else:  # CrossEntropy
        prefix = f"{back_bone}_{device}_lr{learning_rate}_mom{momentum_to_plot}"
        label = "Cross Entropy"

    history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        axes[0].plot(history["val_loss"], label=f"Loss={label}")
    else:
        print(f"File not found: {history_path}")

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation Loss")
axes[0].grid(True)
axes[0].legend()

# ---------- Validation Accuracy ----------
axes[1].set_title(f"Validation Accuracy (backbone={back_bone}, momentum={momentum_to_plot}, DA={data_augments_mode})")

for loss in loss_functions:
    if loss == "FocalLoss":
        prefix = f"{back_bone}_{data_augments_mode}_FocalLoss_{device}_lr{learning_rate}_mom{momentum_to_plot}"
        label = "Focal Loss"
    else:  # CrossEntropy
        prefix = f"{back_bone}_{device}_lr{learning_rate}_mom{momentum_to_plot}"
        label = "Cross Entropy"

    history_path = os.path.join(data_path, f"{prefix}_history_latest.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        axes[1].plot(history["val_acc"], label=f"Loss={label}")
    else:
        print(f"File not found: {history_path}")

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation Accuracy")
axes[1].grid(True)
axes[1].legend()

plt.suptitle(f"Validation Metrics for Different Loss Functions\n(backbone={back_bone}, momentum={momentum_to_plot})", fontsize=14)
plt.tight_layout()
plt.show()
# ---------- End ----------
















