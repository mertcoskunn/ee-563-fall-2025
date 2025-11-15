import time
import torch
from ModelYOLO import BalloonYOLO 

def measure_yolo_inference(model, image_paths, device_type='cpu', conf=0.25, repeat=1):
    device = torch.device(device_type)
    model.model.to(device)

    total_time = 0.0
    count = 0

    for img_path in image_paths:
        for _ in range(repeat):
            start = time.time()
            results = model.predict(img_path)
            if device_type=='cuda':
                torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)
            count += 1

    avg_time = total_time / count
    print(f"YOLO {device_type.upper()} average inference time (repeat={repeat}): {avg_time:.4f} s")
    return avg_time



if __name__ == "__main__":
        yolo_model = BalloonYOLO(None)
        yolo_model.load_model("balloon_detection/yolo_balloon/weights/best.pt")
        image_list = ["balloon1.jpg", "balloon2.jpg", "balloon3.jpg"]
        
        avg_yolo_cpu = measure_yolo_inference(yolo_model, image_list, device_type='cpu', repeat=10)
        if torch.cuda.is_available():
            avg_yolo_gpu = measure_yolo_inference(yolo_model, image_list, device_type='cuda', repeat=10)

        print("Avg cpu: " + str(avg_yolo_cpu))
        print("Avg gpu: " + str(avg_yolo_gpu))