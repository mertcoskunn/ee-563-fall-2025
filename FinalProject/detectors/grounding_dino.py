import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDinoDetector:
    def __init__(
        self,
        model_id="IDEA-Research/grounding-dino-base",
        device=None,
        box_threshold=0.2,
        text_threshold=0.3
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(self.device)
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, image: Image.Image, text_labels):
        inputs = self.processor(
            images=image,
            text=text_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )

        return results[0]

    def detect_and_crop(
        self,
        image_path,
        text_labels,
        save_dir=None,
        draw_debug=False
    ):
        image = Image.open(image_path).convert("RGB")
        image_for_crop = image.copy()

        result = self.detect(image, text_labels)

        crops = []

        for idx, (box, score, label) in enumerate(
            zip(result["boxes"], result["scores"], result["labels"])
        ):
            
            if label is None or str(label).strip() == "":
                continue
            
            x0, y0, x1, y1 = map(int, box.tolist())
            crop = image_for_crop.crop((x0, y0, x1, y1))

            crop_info = {
                "label": label,
                "score": float(score),
                "bbox": (x0, y0, x1, y1),
                "image": crop
            }

            if save_dir:
                filename = f"{label}_{idx}_{score:.2f}.png"
                crop.save(f"{save_dir}/{filename}")
                crop_info["path"] = f"{save_dir}/{filename}"

            crops.append(crop_info)

            if draw_debug:
                self._draw_box(image, x0, y0, x1, y1, label, score)

        return crops, image if draw_debug else crops

    def _draw_box(self, image, x0, y0, x1, y1, label, score):
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        text = f"{label} ({score:.2f})"
        text_box = draw.textbbox((x0, y0), text, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((x0, y0), text, fill="white", font=font)
