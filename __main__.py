import hydra
from omegaconf import DictConfig

from src.app import App
from src.utils.logger import logger

@hydra.main(version_base="1.3", config_path="configs", config_name="app.yaml")
def main(cfg: DictConfig) -> None:
    logger.debug(f"Running app with config: {cfg}")
    app = App(cfg)
    app.run()


if __name__ == "__main__":
    main()