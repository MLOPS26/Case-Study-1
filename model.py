from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch


def setup_model(model_name: str) -> tuple:
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    return model, processor
