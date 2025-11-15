import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt


class BalloonYOLOTrainer:
    def __init__(self, dataset_dir, model_name='yolov8n.pt', project_name='balloon_detection'):
        self.dataset_dir = dataset_dir  
        self.model_name = model_name
        self.project_name = project_name
        self.model = None

    def _convert_via_to_yolo(self, subset):
        
        subset_dir = os.path.join(self.dataset_dir, subset)
        json_path = os.path.join(subset_dir, 'via_region_data.json')

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        images_dir = os.path.join(subset_dir, 'images')
        labels_dir = os.path.join(subset_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for item in tqdm(annotations.values(), desc=f"Converting {subset} set"):
            filename = item['filename']
            img_path = os.path.join(subset_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  Img not found: {img_path}")
                continue

            dest_img_path = os.path.join(images_dir, filename)
            if not os.path.exists(dest_img_path):
                os.rename(img_path, dest_img_path)

            h, w = img.shape[:2]

            yolo_lines = []

            regions = item.get('regions', [])
            if isinstance(regions, dict):
                regions = regions.values()

            for region in regions:
                shape = region['shape_attributes']
                if shape['name'] != 'polygon':
                    continue

                all_x = shape['all_points_x']
                all_y = shape['all_points_y']

                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)

                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                bbox_w = (xmax - xmin) / w
                bbox_h = (ymax - ymin) / h

                yolo_lines.append(f"0 {x_center} {y_center} {bbox_w} {bbox_h}\n")

            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.writelines(yolo_lines)


    def train(self, epochs=50, img_size=640):
      
        self._convert_via_to_yolo('train')
        self._convert_via_to_yolo('val')

        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"""
train: {os.path.abspath(os.path.join(self.dataset_dir, 'train'))}
val: {os.path.abspath(os.path.join(self.dataset_dir, 'val'))}

nc: 1
names: ['balloon']
""")

        self.model = YOLO(self.model_name)

        self.model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            project=self.project_name,
            name='yolo_balloon'
        )



class BalloonYOLO:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.model = None

    def load_model(self, model_path):
       
        self.model = YOLO(model_path)
        print(f"Model loaded: {model_path}")

    def predict(self, image_path, conf=0.25):
        
        if self.model is None:
            raise ValueError("Model didnt load. First run load_model().")

        results = self.model.predict(source=image_path, conf=conf, verbose=False)

        detections = []
        for r in results:  
            boxes = r.boxes  
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()  
                conf_score = boxes.conf[i].item()

                w = x2 - x1
                h = y2 - y1
                x_center = x1 + w/2
                y_center = y1 + h/2

                detections.append([cls, x_center, y_center, w, h, conf_score])

        return detections

    def predict_and_draw(self, image_path, conf=0.25):
        
        if self.model is None:
            raise ValueError("Model couldnt loaded. First run load_model().")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        results = self.model.predict(source=image_path, conf=conf, verbose=False)

        total_balloon_area = 0
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                w_box = x2 - x1
                h_box = y2 - y1
                total_balloon_area += w_box * h_box

                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                conf_score = boxes.conf[i].item()
                cv2.putText(img_rgb, f"{conf_score:.2f}", (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        total_image_area = (h_img * w_img)/2
        balloon_area_percent = (total_balloon_area / total_image_area) * 100

        plt.figure(figsize=(10,10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Balloon area: {balloon_area_percent:.2f}%")
        plt.show()

        return balloon_area_percent

