import torch
from transformers import pipeline
from PIL import Image


class BlipVQA:
    def __init__(
        self,
        model_id="Salesforce/blip-vqa-base",
        device=0
    ):
        self.pipe = pipeline(
            task="visual-question-answering",
            model=model_id,
            device=device
        )

    def ask(self, image: Image.Image, question: str):
        result = self.pipe(
            question=question,
            image=image
        )
        return result[0]["answer"]
