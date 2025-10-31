# print available VRAM

import os, sys
import torch
from transformers import AutoModelForCausalLM

print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import MODEL_IDS

for model_name, model_id in MODEL_IDS.items():
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Loaded model: {model_name}")
    # print(f"Model ID: {model_id}")
    # print(f"Model Name: {model_name}")
    # print(f"Model Config: {model.config}")
    # print(f"Model State Dict: {model.state_dict()}")
    print(f"Model Parameters: {model.num_parameters()}")
    print(f"Model Memory: {model.get_memory_footprint()}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("-"*100)
    del model
    torch.cuda.empty_cache()