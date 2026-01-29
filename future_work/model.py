from unsloth import FastVisionModel
import torch
from consts import BASE_MODEL


def setup_model(model: str) -> tuple:
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit = True, 
        use_gradient_checkpointing = "True",
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, 
        finetune_language_layers   = True, 
        finetune_attention_modules = True, 
        finetune_mlp_modules       = True,

        r = 16, 
        lora_alpha = 16,  
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False, 
        loftq_config = None, 
        use_gradient_checkpointing = "True" 
    )

    return model, tokenizer
