from datasets import load_dataset
from train import train
from reward import formatting_reward_func, correctness_reward_func
from future_work.inference import inference
from future_work.model import setup_model
from future_work.dataset import dataset_setup
from future_work.adapters import save_model, save_gguf
from consts import BASE_MODEL, TRAIN_DATASET
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--idx", type=int, help='what index to test inference on')
    parser.add_argument("--save_model", type=bool, default = True)
    parser.add_argument("--save_gguf", type = bool, default = True)
    parser.add_argument("--local", type=bool, default = True)
    parser.add_argument("--model_name", type=str, default = "lora_model")


    args = parser.parse_args()

    model, tokenizer = setup_model(BASE_MODEL)
    dataset = load_dataset(TRAIN_DATASET, split="testmini")
    train_ds, ds = dataset_setup(dataset, tokenizer)
    reward_fns = [formatting_reward_func, correctness_reward_func]
    trainer = train(tokenizer, model, reward_fns, train_ds)
    eval = inference(args.idx, model, dataset, tokenizer)
    if args.save_model and args.local:
        save_model(model, tokenizer, args.local)
    if args.save_gguf and args.local:
        if not args.model_name:
            save_gguf("math_finetune", args.local, tokenizer)
        else:
            save_gguf(args.model_name, args.local, tokenizer)

