# src/nlp_module.py

from transformers import  pipeline, WhisperProcessor, WhisperForConditionalGeneration, AutoImageProcessor, AutoModelForImageClassification
import torchaudio
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class TransformerToolkit:
    def __init__(self, device="cpu"):
        """
        Initializes the toolkit.
        device: 'cpu' or 'cuda'
        """
        self.device = device

    def sentiment_analysis(self, text, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Performs sentiment analysis on the given text.
        Parameters:
            text (str): Input text for classification
            model_name (str): Hugging Face model to load
        Returns:
            Model prediction result (label and score)
        """
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1 if self.device == "cpu" else 0
        )
        r = classifier(text)
        return r[0]["label"]
    
    def zero_shot_classification(
        self, 
        text, 
        candidate_labels, 
        model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    ):
        """
        Performs zero-shot classification.
        Parameters:
            text (str): input text
            candidate_labels (list): possible labels
            model_name (str): HF model name
        Returns:
            classification scores for each label
        """
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device == "gpu" else -1
        )
        r = classifier(text, candidate_labels)
        return r["labels"][r["scores"].index(max(r["scores"]))]



    def text_generation(
        self, 
        prompt, 
        max_length=50,
        model_name="gpt2"
    ):
        """
        Generates text continuation from an incomplete sentence.
        Parameters:
            prompt (str): starting text
            max_length (int): max length of generated output
            model_name (str): HF language model
        Returns:
            Generated text
        """
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if self.device == "gpu" else -1
        )
        r = generator(prompt,  max_new_tokens=max_length)
        return r[0]["generated_text"]


    def mask_filling(
        self, 
        text, 
        model_name="bert-base-uncased"
    ):
        """
        Predicts the missing token in a masked sentence.
        Parameters:
            text (str): sentence containing [MASK]
            model_name (str): HF masked language model
        Returns:
            List of predictions for the mask
        """
        filler = pipeline(
            "fill-mask",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        r = filler(text)
        return r[0]["sequence"]


    def question_answering(
        self, 
        question, 
        context, 
        model_name="distilbert-base-cased-distilled-squad"
    ):
        """
        Answers a question given a context paragraph.
        Parameters:
            question (str): question text
            context (str): supporting paragraph
            model_name (str): HF QA model
        Returns:
            Extracted answer
        """
        qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            device=0 if self.device == "gpu" else -1
        )

        r = qa_pipeline({ "question": question, "context": context})
        return r["answer"]

    def text_summarization(self, text, max_length=100, min_length=30):
        """Summarize long text into a shorter version."""
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device = "cuda" if self.device == "gpu" else "cpu"
        )
        result = summarizer(text, max_length=max_length, min_length=min_length)
        
        return result[0]["summary_text"]
    

    def text_translation(self, text, source_lang="en", target_lang="tr"):
        """
        Translates text between English and Turkish.
        Supported directions:
            en -> tr
            tr -> en
        """
        if source_lang == "en" and target_lang == "tr":
            model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
        elif source_lang == "tr" and target_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-tr-en"
        else:
            raise ValueError("Only en<->tr translation is supported.")

        translator = pipeline(
            "translation",
            model=model_name,
            device=0 if self.device == "gpu" else -1
        )

        result = translator(text)
        return result[0]["translation_text"]



   
    def named_entity_recognition(self, text):
        """Detect named entities such as persons, organizations, and locations."""
         
        ner = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=self.device if self.device != "gpu" else "cuda"
        )

        entities = ner("Elon Musk founded Tesla in California")
        all_words = [entity['word'] for entity in entities]

        return all_words
    
    
    def image_classification(self, image_path, model_name="google/vit-base-patch16-224"):
        """
        Performs image classification on the given image.
        Parameters:
            image_path (str): Path to the image file
            model_name (str): Hugging Face image classification model
        Returns:
            Predicted class label
        """

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        device = "cuda" if self.device == "gpu" else "cpu"
        model = model.to(device)

        img = Image.open(image_path).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_id = logits.argmax(-1).item()
        return model.config.id2label[pred_id]
    

    def speech_recognition(self, audio_path, model_name="openai/whisper-small"):

        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

        device = "cuda" if self.device in ["cuda", "gpu"] else "cpu"
        model = model.to(device)

        audio, sr = torchaudio.load(audio_path)

        # stereo → mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # resample
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        audio = audio.squeeze()

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

        with torch.no_grad():
            predicted_ids = model.generate(**inputs)

        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    def generate_image(
        self, 
        prompt: str, 
        width: int = 512, 
        height: int = 512, 
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5, 
        save_path: str = "generated_image.png"
    ):
        """
        Generates an image from a text prompt using Stable Diffusion.
        Parameters:
            prompt (str): Text prompt
            width (int): Image width
            height (int): Image height
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Classifier-free guidance scale
            save_path (str): Path to save generated image
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        pipe = pipe.to(device)

        image = pipe(
            prompt, 
            width=width, 
            height=height, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale
        ).images[0]

        image.save(save_path)
        return save_path



if __name__ == "__main__":
    toolkit = TransformerToolkit(device="gpu")

    #output = toolkit.sentiment_analysis("I love this course!")
    #print(output)

    ##print("\nText generation:")
    ##print(toolkit.text_generation("The future of AI is"))

    ##print("\nMask filling:")
    ##print(toolkit.mask_filling("The capital of France is [MASK]."))

    ##print("\nQuestion Answering:")
    ##print(toolkit.question_answering(
    ##    "Who wrote The Lord of the Rings?",
    ##    "The Lord of the Rings is a novel written by J.R.R. Tolkien."
    ##))

    ## ---- Named Entity Recognition ----
    ##print(toolkit.named_entity_recognition(
    ##    "Elon Musk founded SpaceX in the United States."
    ##))

    ##print("\nZero-shot:")
    ##print(toolkit.zero_shot_classification(
    ##    "This movie is amazing",
    ##    ["positive", "negative", "neutral"]
    ##))

    ##print("\n=== Image Classification ===")
    ##img_path = "test.jpg"   # kendi fotoğraf yolunu koy
    ##result = toolkit.image_classification(img_path)
    ##print(result)

    ##print("\n=== Speech Recognition ===")
    ##audio_path = "speech.wav"
    ##result = toolkit.speech_recognition(audio_path)
    ##print(result)

    

    ## ---- Summarization ----
    ##print(toolkit.text_summarization(
    ##    "Artificial intelligence has rapidly evolved in recent years..."
    ##))

    # ---- Translation ----
    #print(toolkit.text_translation(
    #    "today there is rain", 
    #))

    save_path = toolkit.generate_image(
    prompt="A futuristic city at sunset, cyberpunk style",
    width=512,
    height=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    save_path="cyberpunk_city.png"
        )
    print(f"Image saved at: {save_path}")

    

    

    
