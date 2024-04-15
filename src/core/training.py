import hydra
from pathlib import Path

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model #, AutoPeftModelForCausalLM

from trl import SFTTrainer

from omegaconf import DictConfig

from src.utils.instruction import format_instruction
from src.utils.logger import logger

class FinetuningTraining():
    def __init__(self, cfg: DictConfig) -> None:
        logger.debug(f"Initializing FinetuningTraining...")
        dataset = hydra.utils.instantiate(cfg.data)
        logger.debug(f"Dataset loaded...")
        model = hydra.utils.instantiate(cfg.model.model)
        logger.debug(f"Model loaded...")
        tokenizer = hydra.utils.instantiate(cfg.model.tokenizer)
        logger.debug(f"Tokenizer loaded...")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
            ]
        )
        logger.debug(f"Peft config loaded...")

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        logger.debug(f"Model prepared for kbit training...")

        self.output_dir = Path(cfg.paths.output_dir)
        self.log_dir = Path(cfg.paths.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output and log directories created...")

        training_args = hydra.utils.instantiate(cfg.trainer.train_config)
        logger.debug(f"Training arguments loaded...")

        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=cfg.trainer.max_seq_len,
            tokenizer=tokenizer,
            packing=cfg.trainer.packing,
            formatting_func=format_instruction,
            args=training_args,
            neftune_noise_alpha=cfg.trainer.neftune_noise_alpha,
        )
        logger.debug(f"Trainer initialized...")

    def run(self):
        logger.debug(f"Training model...")
        self.trainer.train()
        logger.debug(f"Training complete...")

        logger.debug(f"Saving model...")
        self.trainer.save_model()
        logger.debug(f"Model saved...")

