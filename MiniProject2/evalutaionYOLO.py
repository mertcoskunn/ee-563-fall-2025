import os
import glob
import numpy as np
from tqdm import tqdm
import cv2

def compute_iou(boxA, boxB):
    """ box format: [x1, y1, x2, y2] """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    unionArea = areaA + areaB - interArea + 1e-6
    return interArea / unionArea


def load_gt_boxes(label_path, img_w, img_h):
    """ YOLO txt formatından GT bbox yükle """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def compute_ap_50(model, val_images_dir, val_labels_dir, conf=0.25):
    """
    AP@0.5 hesaplar.
    model: YOLO model instance
    val_images_dir: validation images/ klasörü
    val_labels_dir: validation labels/ klasörü
    """

    detections = []  # (confidence, TP_or_FP)
    total_gt_boxes = 0

    image_paths = glob.glob(os.path.join(val_images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(val_images_dir, "*.png"))

    for img_path in tqdm(image_paths, desc="Evaluating"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # GT yükle
        label_path = os.path.join(
            val_labels_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        gt_boxes = load_gt_boxes(label_path, w, h)
        total_gt_boxes += len(gt_boxes)
        matched = [False] * len(gt_boxes)

        # Tahminler
        results = model.predict(img_path, conf=conf, verbose=False)
        if len(results) == 0:
            continue

        preds = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append((box.conf.item(), [x1, y1, x2, y2]))

        preds = sorted(preds, key=lambda x: x[0], reverse=True)

        # Tahminleri GT ile eşleştir
        for score, pred_box in preds:
            best_iou = 0
            best_idx = -1

            for i, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= 0.5 and not matched[best_idx]:
                detections.append((score, 1))  # TP
                matched[best_idx] = True
            else:
                detections.append((score, 0))  # FP

    # Precision–Recall hesaplama
    detections = sorted(detections, key=lambda x: x[0], reverse=True)

    TP = 0
    FP = 0
    precisions = []
    recalls = []

    for score, is_tp in detections:
        if is_tp:
            TP += 1
        else:
            FP += 1

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (total_gt_boxes + 1e-6)

        precisions.append(precision)
        recalls.append(recall)

    # AP: PR eğrisi altında alan (numerik integral)
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += precisions[i] * (recalls[i] - recalls[i-1])

    return ap
