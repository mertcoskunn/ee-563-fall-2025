import os
import json
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_scale_from_bbox(crop,floor_width=10.0):
    """
    floor_width: Blender sahnesindeki zemin genişliği (unit)
    """
    x_min, y_min, x_max, y_max = crop["bbox"]

    W_img, H_img = crop["image_size"] 

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    w_norm = bbox_w / W_img
    h_norm = bbox_h / H_img

    bbox_norm_size = max(w_norm, h_norm)

    target_3d_size = bbox_norm_size * floor_width

    return {
        "bbox_norm_size": bbox_norm_size,
        "target_3d_size": target_3d_size
    }

def compute_x_position_from_bbox(crop, floor_width=10.0):
    """
    Sadece sağ-sol (X) konumu hesaplar.
    """
    x_min, y_min, x_max, y_max = crop["bbox"]
    W_img, H_img = crop["image_size"]

    bbox_center_x = (x_min + x_max) / 2.0
    image_center_x = W_img / 2.0

    # [-1, +1] aralığı
    x_norm = (bbox_center_x - image_center_x) / (W_img / 2.0)

    # Blender world coordinate
    x_world = x_norm * (floor_width / 2.0)

    return {
        "x_norm": x_norm,
        "x_world": x_world
    }

def compute_depth_from_bbox(crop, floor_depth=10.0):
    """
    Bounding box merkezine göre derinlik (Z ekseni) hesaplar.
    Görüntü merkezi = 0
    Alt → pozitif (öne)
    Üst → negatif (arkaya)
    """

    x_min, y_min, x_max, y_max = crop["bbox"]
    W_img, H_img = crop["image_size"]

    bbox_center_y = (y_min + y_max) / 2.0
    image_center_y = H_img / 2.0

    # [-1, +1] aralığı
    z_norm = (bbox_center_y - image_center_y) / (H_img / 2.0)

    # Blender world Z coordinate
    z_world = z_norm * (floor_depth / 2.0)

    return {
        "z_norm": z_norm,
        "z_world": z_world
    }

def compute_wall_height_from_bbox(crop, wall_height=3.0):
    """
    Bounding box merkezine göre duvar üzerindeki yükseklik hesaplar.
    Görüntü merkezi = 0
    Üst → pozitif
    Alt → negatif
    """

    x_min, y_min, x_max, y_max = crop["bbox"]
    W_img, H_img = crop["image_size"]

    bbox_center_y = (y_min + y_max) / 2.0
    image_center_y = H_img / 2.0

    # [-1, +1] aralığı
    y_norm = (image_center_y - bbox_center_y) / (H_img / 2.0)

    # Blender world vertical coordinate (örneğin Y ekseni)
    y_world = y_norm * (wall_height / 2.0)

    return {
        "y_norm": y_norm,
        "y_world": y_world
    }

def save_scene_from_crops(crops, wall_color, floor_color, scene_path="scene.json"):
    objects = []

    for crop in crops:
        if "meshes" not in crop:
             continue

        mesh = crop["meshes"][0] 

        objects.append({
            "obj_path": mesh["obj"],
            "ply_path": mesh["ply"],
            "label": crop["label"],
            "scale": crop["scale"]["target_3d_size"],
            "placement": crop["placement"],
            "location": {
                "x": crop["depth"]["z_world"],
                "y": crop["position"]["x_world"],
                "z": crop["height"]["y_world"]
            }
        })

    scene_data = {
        "floor": {
            "color": floor_color  # [r, g, b] in 0–1
        },
        "wall": {
            "color": wall_color   # [r, g, b] in 0–1
        },
        "objects": objects
    }

    with open(scene_path, "w", encoding="utf-8") as f:
        json.dump(scene_data, f, indent=2)

def estimate_wall_floor_colors(image_path,
                               wall_y_ratio=0.25,
                               floor_y_ratio=0.75):
    """
    Duvar rengi: görüntünün orta-üst kısmından tek pixel
    Zemin rengi: görüntünün orta-alt kısmından tek pixel

    wall_y_ratio: duvar için yükseklik oranı (0–1)
    floor_y_ratio: zemin için yükseklik oranı (0–1)
    """
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    H, W, _ = img_np.shape
    x_mid = W // 2

    wall_y = int(H * wall_y_ratio)
    floor_y = int(H * floor_y_ratio)

    wall_color = img_np[wall_y, x_mid].tolist()
    floor_color = img_np[floor_y, x_mid].tolist()

    return {
        "wall_color": wall_color,   # [R, G, B]
        "floor_color": floor_color  # [R, G, B]
    }