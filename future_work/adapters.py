from transformers import TextStreamer
from unsloth import FastVisionModel
from dotenv import load_dotenv
import os


def save_model(model, tokenizer, local: bool) -> None:
    load_dotenv()
    if local:
        model.save_pretrained()
        tokenizer.save_pretrained("ft_llava")
    else:
        model.push_to_hub(f"{os.getenv("ORG_NAME")}/ft_llava", token = os.getenv("HF_TOKEN"))        
    return 


def save_gguf(model_name: str, local:bool, tokenizer): 
    model, processor = FastVisionModel.from_pretrained(
        model_name= model_name, 
        load_in_4bit=True,  
    )
    FastVisionModel.for_inference(model) 
    if local:
        model.save_pretrained_merged("ft_qwen2_vl_2b", tokenizer)

    else:
        model.push_to_hub_merged(f"{os.getenv("ORG_NAME")}/ft_qwen2_vl_2b", tokenizer, token = f"{os.getenv("HF_TOKEN")}")
