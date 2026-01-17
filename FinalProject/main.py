from detectors.grounding_dino import GroundingDinoDetector
from captioners.blip_captioner import BlipVQA
from utils.image_utils import ensure_dir, compute_scale_from_bbox, compute_x_position_from_bbox, compute_depth_from_bbox, compute_wall_height_from_bbox, save_scene_from_crops, estimate_wall_floor_colors
from generator.mesh_generator import ShapeEMeshGenerator
import subprocess
import os
from PIL import Image

PLACEMENT_RULES = {
    "couch": "floor",
    "armchair": "floor",
    "coffee table": "floor",
    "table": "floor",
    "plant": "floor",
    "table lamp": "floor",
    "bed": "floor",
    "cupboard": "floor",

    "painting": "wall",
    "picture": "wall",

    "ceiling lamp": "ceiling",
    "lamp": "ceiling"
}



def save_captions(crops, file_path="captions.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, crop in enumerate(crops):
            caption = crop.get("caption")
            if not caption:
                continue
            f.write(f"{i}: {caption}\n")

IMAGE_PATH = "test_image.png"
CROP_DIR = "crops"

with Image.open(IMAGE_PATH) as img:
    IMG_SIZE = img.size 

CLASSES = [["armchair", "lamp", "table lamp", "couch", "table", "painting", "plant", "bed", "cupboard"]]

ensure_dir(CROP_DIR)

detector = GroundingDinoDetector(
    box_threshold=0.2,
    text_threshold=0.5
)

crops, debug_image = detector.detect_and_crop(
    image_path=IMAGE_PATH,
    text_labels=CLASSES,
    save_dir=CROP_DIR,
    draw_debug=True
)

debug_image.save("detected_image.png")


# ---------- BLOCK 2: BLIP ----------
vqa = BlipVQA(device=0)

for crop in crops:
    cls = crop["label"]  
    crop["placement"] = PLACEMENT_RULES.get(crop["label"])
    color = vqa.ask(
        crop["image"],
        f"What is the color of the {cls}?"
    )
    crop["caption"] = f"a {color} {cls}"
    crop["image_size"] = IMG_SIZE
    print(crop["caption"])
    
save_captions(crops, "captions.txt")


print("#############################33")
shape_e = ShapeEMeshGenerator()

ensure_dir("output")

all_meshes = []

for i, crop in enumerate(crops):
    
    caption = crop.get("caption")
    if not caption:
        continue

    safe_label = crop["label"].replace(" ", "_")
    output_prefix = f"output/{safe_label}_{i}"

    print(f"Generating mesh for: {caption}")

    meshes = shape_e.generate_meshes(
        prompt=caption,
        batch_size=1,
        output_prefix=output_prefix
    )

    crop["meshes"] = meshes
    all_meshes.append(meshes)

print("All generated meshes:", all_meshes)

for crop in crops:
    crop["scale"] = compute_scale_from_bbox(crop, floor_width=10.0)
    if(crop["placement"] == "floor"): 
        crop["position"] = compute_x_position_from_bbox(crop, floor_width=10.0)
        crop["depth"] = compute_depth_from_bbox(crop, floor_depth=10.0)
        crop["height"] = {"y_world": 0.0, "y_norm":0.0}
    
    elif(crop["placement"] == "wall"):
        crop["position"] = compute_x_position_from_bbox(crop, floor_width=10.0)
        crop["depth"]={"z_world": -9.0, "z_norm":-9.0}
        crop["height"] = compute_wall_height_from_bbox(crop, wall_height=3.0)
    
    else:
        crop["position"] = {"x_world": 0.0, "x_norm":0.0}
        crop["depth"]= {"z_world": 0.0, "z_norm":0.0}
        crop["height"] = {"y_world": 3.0, "y_norm":3.0}

colors = estimate_wall_floor_colors(IMAGE_PATH)
wall_color = colors["wall_color"]
floor_color = colors["floor_color"]
save_scene_from_crops(crops, wall_color, floor_color, scene_path="scene.json")


BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
BLENDER_SCRIPT = os.path.abspath("show_in_blender.py")


subprocess.run([
    BLENDER_PATH,
    "--python", "show_in_blender.py",
    "--",
    os.path.abspath("scene.json")
])