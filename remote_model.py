from huggingface_hub import InferenceClient
import huggingface_hub
from consts import BASE_MODEL
from PIL import Image
from transformers import pipeline
import time


pipe = pipeline("image-text-to-text", model = BASE_MODEL)


def query_remote(image: Image.Image, question: str, pipe):
    start_time = time.time()
    print("starting remote inference... %s" %(start_time))
    if not Image:
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

    outputs = pipe(text=messages, return_full_text=False)

    print("remote time %s --- " % (time.time() - start_time))

    return outputs[0]["generated_text"]
