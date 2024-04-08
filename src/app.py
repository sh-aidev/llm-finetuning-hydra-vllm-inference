from typing import Tuple, Dict, List

import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from peft import PeftModel
from trl import SFTTrainer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path
from datasets import load_dataset

@hydra.main(version_base="1.3", config_path="../configs", config_name="app.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    # data = hydra.utils.instantiate(cfg.data)
    # cfg.model.bitsandbytes.bnb_4bit_compute_dtype = torch.bfloat16
    # cfg["model"]["model"]["quantization_config"]["bnb_4bit_compute_dtype"] = torch.bfloat16
    # print(cfg)
    # model = hydra.utils.instantiate(cfg.model.model)
    # tokenizer = hydra.utils.instantiate(cfg.model.tokenizer)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    # peft_config = hydra.utils.instantiate(cfg.model.lora)

if __name__ == "__main__":
    main()