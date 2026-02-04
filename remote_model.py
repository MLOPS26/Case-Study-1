from huggingface_hub import InferenceClient
from PIL import Image
import time
import base64
from io import BytesIO

remote_model="zai-org/GLM-4.5V"

client = InferenceClient(model=remote_model)

def query_remote(image: Image.Image, question: str, client: InferenceClient):
    start_time = time.time()
    print("starting remote inference... %s" %(start_time))

    if not Image:
        raise ValueError("Missing image")

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{img_str}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": question}
            ]
        }
    ]

    response = client.chat.completions.create(messages=messages, max_tokens=256)

    print("remote time %s --- " % (time.time() - start_time))

    return response.choices[0].message.content
