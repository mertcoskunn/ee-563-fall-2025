import gradio as gr
from src.nlp_module import TransformerToolkit

toolkit = TransformerToolkit(device="gpu")


def sentiment_fn(text):
    return toolkit.sentiment_analysis(text)

def qa_fn(context, question):
    return toolkit.question_answering(context, question)

def zero_shot_fn(text, labels):
    label_list = [l.strip() for l in labels.split(",")]
    return toolkit.zero_shot_classification(text, label_list)

def summary_fn(text):
    return toolkit.text_summarization(text)

def gen_fn(prompt, length):
    return toolkit.text_generation(prompt, max_length=int(length))

def translate_fn(text, direction):
    if direction == "English → Turkish":
        return toolkit.text_translation(text, source_lang="en", target_lang="tr")
    else:
        return toolkit.text_translation(text, source_lang="tr", target_lang="en")

def mask_fn(text):
    return toolkit.mask_filling(text)

def img_fn(image):
    return toolkit.image_classification(image)

def ner_fn(text):
    return toolkit.named_entity_recognition(text)

def asr_fn(audio_path):
    return toolkit.speech_recognition(audio_path)

def image_generator_fn(prompt, width, height, num_inference_steps, guidance_scale, save_path):
    return toolkit.generate_image(prompt=prompt,
        width=int(width),
        height=int(height),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        save_path=save_path)


# ==============================
#  GRADIO UI
# ==============================

with gr.Blocks(title="EE563 Mini Project 3 Demo") as demo:

    gr.Markdown("## EE563 – Transformers Demo (All tasks)")

    # ---- Sentiment ----
    with gr.Tab("Sentiment Analysis"):
        inp = gr.Textbox(label="Input Text")
        out = gr.Textbox(label="Sentiment Result")
        btn = gr.Button("Run")
        btn.click(sentiment_fn, inp, out)

    # ---- QA ----
    with gr.Tab("Question Answering"):
        ctx = gr.Textbox(label="Context Paragraph", lines=4)
        q = gr.Textbox(label="Question")
        ans = gr.Textbox(label="Answer")
        gr.Button("Run").click(qa_fn, [q, ctx], ans)

    # ---- Zero Shot ----
    with gr.Tab("Zero Shot Classification"):
        zt = gr.Textbox(label="Text")
        zl = gr.Textbox(label="Candidate Labels (comma-separated)")
        zo = gr.Textbox(label="Result")
        gr.Button("Run").click(zero_shot_fn, [zt, zl], zo)

    # ---- Summarization ----
    with gr.Tab("Summarization"):
        sm_in = gr.Textbox(label="Text to summarize", lines=4)
        sm_out = gr.Textbox(label="Summary")
        gr.Button("Run").click(summary_fn, sm_in, sm_out)
    
    with gr.Tab("Named Entity Recognition"):
        sm_in = gr.Textbox(label="Input text")
        sm_out = gr.Textbox(label="Output")
        gr.Button("Run").click(ner_fn, sm_in, sm_out)

    # ---- Text Generation ----
    with gr.Tab("Text Generation"):
        gen_in = gr.Textbox(label="Prompt text")
        gen_len = gr.Number(label="Max length", value=50)
        gen_out = gr.Textbox(label="Generated Text", lines = 4)
        gr.Button("Run").click(gen_fn, [gen_in, gen_len], gen_out)

    # ---- Translation ----
    with gr.Tab("Translation"):
        tr_text = gr.Textbox(label="Input Text")
        direction = gr.Radio(
            ["English → Turkish", "Turkish → English"],
            label="Direction",
            value="English → Turkish"
        )
        tr_out = gr.Textbox(label="Translated Text")
        gr.Button("Translate").click(translate_fn, [tr_text, direction], tr_out)

    # ---- Mask Filling ----
    with gr.Tab("Mask Filling"):
        mf_in = gr.Textbox(label="Text containing [MASK]")
        mf_out = gr.Textbox(label="Predictions")
        gr.Button("Run").click(mask_fn, mf_in, mf_out)

    # ---- Image Classification ----
    with gr.Tab("Image Classification"):
        img_in = gr.Image(type="filepath", label="Upload Image")
        img_out = gr.Textbox(label="Class")
        gr.Button("Run").click(img_fn, img_in, img_out)

    # ---- Speech Recognition ----
    with gr.Tab("Speech Recognition"):
        audio_in = gr.Audio(type="filepath", label="Upload WAV/MP3")
        audio_out = gr.Textbox(label="Transcription")
        gr.Button("Run").click(asr_fn, audio_in, audio_out)
    
    with gr.Tab("Image Generation"):
        prompt_in = gr.Textbox(label="Prompt", placeholder="Enter image description")
        width_in = gr.Number(label="Width", value=512)
        height_in = gr.Number(label="Height", value=512)
        steps_in = gr.Number(label="Num Inference Steps", value=50)
        guidance_in = gr.Number(label="Guidance Scale", value=7.5)
        save_path_in = gr.Textbox(label="Save Path", placeholder="output.png")

        img_out = gr.Image(type="pil", label="Generated Image")

        gr.Button("Run").click(
            fn=lambda prompt, w, h, steps, scale, path: toolkit.generate_image(
                prompt=prompt,
                width=int(w),
                height=int(h),
                num_inference_steps=int(steps),
                guidance_scale=float(scale),
                save_path=path
            ),
            inputs=[prompt_in, width_in, height_in, steps_in, guidance_in, save_path_in],
            outputs=img_out
        )

demo.launch()
