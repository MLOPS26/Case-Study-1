import torch
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from qwen_vl_utils import process_vision_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, device
    # model = Qwen2VLForConditionalGeneration.from_pretrained(BASE_MODEL)
    # model.to(DEVICE)
    # processor = AutoProcessor.from_pretrained(BASE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield
    model = None
    processor = None
    device = None


app = FastAPI(lifespan=lifespan)


@app.get("/device")
async def get_device():
    print(globals())
    return {"device": device}


@app.post("/inference")
async def inference(question: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    try:
        if processor and model in globals():

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=text, images=images, videos=video_inputs, padding=True, return_tensors="pt"
            )
 
            inputs = inputs.to(device)


            generated_ids = await run_in_threadpool(
                        model.generate, **inputs, max_new_tokens=256
                    )

            generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

            output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
            return {"response": output_text[0]} 
    except Exception as e:
        raise ValueError(f"Missing processor or model  in globs, error: {e}")
   
