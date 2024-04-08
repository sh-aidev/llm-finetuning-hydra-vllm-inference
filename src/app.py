from typing import Tuple, Dict, List
from omegaconf import DictConfig
from src.core.training import FinetuningTraining
from src.utils.logger import logger

class App:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.trainer = FinetuningTraining(cfg)
    
    def run(self):
        self.trainer.train()
        self.trainer.save_model()
        self.trainer.push_to_huggingface()