import torch
import gradio as gr
from PIL import Image
from consts import BASE_MODEL
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2VLForConditionalGeneration.from_pretrained(BASE_MODEL)
processor = AutoProcessor.from_pretrained(BASE_MODEL)


def query_local(image: Image.Image, question: str):
    start_time = time.time()
    print("starting local inference at: %s" %( start_time))
    if not image:
        raise ValueError("Missing image")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    images, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text, 
        images=images, 
        videos=video_inputs,
        padding=True, 
        return_tensors="pt")

    generated_ids = model.generate(**inputs, max_new_tokens=256)

    print("inputs generated")
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print("trimmed")
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    print("decoded")

    print("local %s --- " % (time.time() - start_time))

    return output_text[0]

