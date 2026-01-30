import torch


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
TRAIN_DATASET = "AI4Math/MathVista"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR = None
APP_DB_NAME = "app_users.db"


