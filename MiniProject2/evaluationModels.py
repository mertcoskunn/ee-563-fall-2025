from ModelYOLO import BalloonYOLO
from evalutaionYOLO import compute_ap_50 
from ModelUNET import BallonSegmentator_Unet, BalloonDataset 


model_yolo = BalloonYOLO(dataset_dir="")
model_yolo.load_model("balloon_detection/yolo_balloon/weights/best.pt")

ap50 = compute_ap_50(
   model_yolo.model,
   val_images_dir="balloon/val/images",
   val_labels_dir="balloon/val/labels"
)

print("AP@0.5 score for YOLO= ", ap50)


model_unet = BallonSegmentator_Unet()
model_unet.load_model("unet_balloon.pth")
val_dataset = BalloonDataset("balloon_1/val")

ap50 = model_unet.evaluate_ap50(val_dataset)
print("AP@0.5 score for UNET= ", ap50)