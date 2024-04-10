import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
# from typing import Tuple, Dict, List
from omegaconf import DictConfig
from src.core.training import FinetuningTraining
from src.core.inference import LLMInference, LLMInferenceHF
from src.server.server import LLMServer, get_router
from src.utils.logger import logger

class App:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        if self.cfg.task_name == "train":
            self.llm = FinetuningTraining(cfg)
        elif self.cfg.task_name == "infer":
            self.llm = LLMInference(cfg)
        elif self.cfg.task_name == "server":
            router = get_router(cfg)
            self.llm = LLMServer(cfg, router)

    def run(self):
        if self.cfg.task_name == "infer":
            if self.cfg.push_huggingface == True:
                self.llm.push_to_huggingface()
            INPUT = """
                ### Instruction:
                List 3 historical events related to the following country

                ### Input:
                Canada

                ### Response:
            """
            logger.debug(f"Input: {INPUT}")
            self.llm.run(INPUT)
        else:
            self.llm.run()
