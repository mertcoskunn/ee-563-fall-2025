from detectors.grounding_dino import GroundingDinoDetector
from captioners.blip_captioner import BlipVQA
from utils.image_utils import ensure_dir
from generator.mesh_generator import ShapeEMeshGenerator
import subprocess
import os

def save_captions(crops, file_path="captions.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, crop in enumerate(crops):
            caption = crop.get("caption")
            if not caption:
                continue
            f.write(f"{i}: {caption}\n")

IMAGE_PATH = "test_image.png"
CROP_DIR = "crops"

CLASSES = [["armchair", "lamp", "coffee table", "couch", "table", "painting"]]

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

    color = vqa.ask(
        crop["image"],
        f"What is the color of the {cls}?"
    )

    crop["caption"] = f"a {color} {cls}"

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


BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
BLENDER_SCRIPT = os.path.abspath("show_in_blender.py")


subprocess.run([
    BLENDER_PATH,
    "--python", "show_in_blender.py",
    "--",
    os.path.abspath("output")
])