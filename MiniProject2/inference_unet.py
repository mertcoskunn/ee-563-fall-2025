import torch
import cv2
import time
from ModelUNET import BallonSegmentator_Unet

def measure_unet_inference(model, image_paths, device_type='cpu', resize=(256,256), repeat=1):
    device = torch.device(device_type)
    model.model.to(device)

    total_time = 0.0
    count = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img_rgb, resize)
        input_tensor = torch.tensor(input_img, dtype=torch.float).permute(2,0,1).unsqueeze(0)/255.0
        input_tensor = input_tensor.to(device)

        for _ in range(repeat):
            start = time.time()
            with torch.no_grad():
                pred = model.model(input_tensor)
                if device_type=='cuda':
                    torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)
            count += 1

    avg_time = total_time / count
    print(f"U-Net {device_type.upper()} average inference time (repeat={repeat}): {avg_time:.4f} s")
    return avg_time


if __name__ == "__main__":
        unetModel = BallonSegmentator_Unet()
        unetModel.load_model("unet_balloon.pth")
        image_list = ["balloon1.jpg", "balloon2.jpg", "balloon3.jpg"]
        
        avg_yolo_cpu = measure_unet_inference(unetModel, image_list, device_type='cpu', repeat=10)
        if torch.cuda.is_available():
            avg_yolo_gpu = measure_unet_inference(unetModel, image_list, device_type='cuda', repeat=10)

        print("Avg cpu: " + str(avg_yolo_cpu))
        print("Avg gpu: " + str(avg_yolo_gpu))