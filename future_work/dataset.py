from datasets import Dataset
from consts import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END 



def is_numeric_answer(example):
    try:
        float(example["answer"])
        return True
    except Exception as e:
        return f"error: {e}"

def resize_images(example):
    image = example["decoded_image"]
    image = image.resize((512,512))
    example["decoded_image"] = image
    return example


def convert_to_rgb(example):
    image = example["decoded_image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["decoded_image"] = image
    return example


def make_conversation(example):

    text_content = (
        f"{example['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} "
        f"and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
    )

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": text_content},  # The text part of the prompt
            ],
        },
    ]

    # The actual image data is kept separate for the processor
    return {"prompt": prompt, "image": example["decoded_image"], "answer": example["answer"]}



def dataset_setup(dataset: Dataset, tokenizer) -> Dataset:
    dataset = dataset.filter(is_numeric_answer) 
    dataset = dataset.map(resize_images) 
    dataset = dataset.map(convert_to_rgb)
    train_dataset = dataset.map(make_conversation)

    #We dataset is reformattted like this because decoded_images are the actual images (since we are in the minitest split)
    #The "image": example["decoded_image"] does not properly format the dataset correctly
    train_dataset = train_dataset.remove_columns("image")
    train_dataset = train_dataset.rename_column("decoded_image", "image")

    train_dataset = train_dataset.map(
        lambda example: {
            "prompt": tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=False
            )
        }
    )
    return train_dataset, dataset

