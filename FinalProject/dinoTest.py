import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = "hh.png"
image = Image.open(image_path).convert("RGB")
image_for_crop = image.copy()   # <-- temiz kopya

text_labels = [["armchair, lamp, coffe table, couch, paint, furniture, table, stand"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.2,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# ---------- DRAW DETECTIONS ----------
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

result = results[0]

for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    x0, y0, x1, y1 = box.tolist()

    # bounding box
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

    # label text
    text = f"{label} ({score:.2f})"
    text_size = draw.textbbox((x0, y0), text, font=font)

    draw.rectangle(
        [text_size[0], text_size[1], text_size[2], text_size[3]],
        fill="red"
    )
    draw.text((x0, y0), text, fill="white", font=font)

image.show()
image.save("detected_image.png")


#import os
#os.makedirs("crops", exist_ok=True)

for idx, (box, score, label) in enumerate(
    zip(result["boxes"], result["scores"], result["labels"])
):
    x0, y0, x1, y1 = map(int, box.tolist())

    # --- CROP (NO BOX, NO TEXT) ---
    crop = image_for_crop.crop((x0, y0, x1, y1))
    crop.save(f"crops/{label}_{idx}_{score:.2f}.png")

    # --- DRAW (ONLY FOR VISUALIZATION) ---
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    text = f"{label} ({score:.2f})"
    text_size = draw.textbbox((x0, y0), text, font=font)
    draw.rectangle(text_size, fill="red")
    draw.text((x0, y0), text, fill="white", font=font)