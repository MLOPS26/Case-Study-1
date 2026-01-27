from transformers import TextStreamer, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from dotenv import load_dotenv
import os


def save_model(model, tokenizer, local: bool):
    if local:
        model_name = "lora_model"
        model.save_pretrained(model_name)  # Local saving
        tokenizer.save_pretrained(model_name)
    else:
        raise NotImplementedError
    return model_name


def save_gguf(model_name: str, local: bool, tokenizer):
    load_dotenv()
    # Loading base model and merging with LoRA
    base_model_name = "Qwen/Qwen2-VL-7B-Instruct"  # Assuming this is the base, consider importing from consts if possible
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name, device_map="auto", trust_remote_code=True
    )

    if local:
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.merge_and_unload()

        output_dir = "math_finetune"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Merged model saved to {output_dir}")
        # GGUF conversion usually requires external tools (llama.cpp) or specific libraries not just HF save
        # Standard HF save is performed here as a CPU-compatible equivalent step for 'saving'

    else:
        # model.push_to_hub_merged(...) # PEFT merge_and_unload then push_to_hub
        raise NotImplementedError(
            "Remote push not fully adapted for generic PEFT in this snippet yet"
        )
