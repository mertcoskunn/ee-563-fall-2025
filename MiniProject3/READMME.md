# EE 563 Mini Project 3 - AI in Practice

This repository contains an AI toolkit implemented with Hugging Face Transformers, Diffusers, and Gradio. The toolkit supports multiple NLP, CV, and multimodal tasks such as:

- Sentiment Analysis
- Question Answering
- Zero-shot Classification
- Text Summarization
- Text Generation
- Text Translation
- Mask Filling
- Image Classification
- Named Entity Recognition
- Automatic Speech Recognition
- Text-to-Image Generation

## Requirements

This project is tested with **Python 3.10+**. You can install the required packages using `pip`:

```bash
pip install gradio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # replace cu118 with your CUDA version, or omit for CPU
pip install transformers
pip install diffusers["torch"]
pip install pillow
```

## Run the Gradio demo
```bash
python gradio_demo.py
```
