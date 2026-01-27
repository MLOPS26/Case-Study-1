from consts import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
from transformers import TextStreamer
import torch


def inference(idx: int, model, dataset, processor):
    # FastVisionModel.for_inference(model) # Not needed for standard HF/PEFT
    image = dataset[idx]["decoded_image"]
    instruction = (
        f"{dataset[idx]['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} "
        f"and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    text_streamer = TextStreamer(processor, skip_prompt=True)
    with torch.no_grad():
        result = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
    return result
