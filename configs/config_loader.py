import os
import yaml

from easydict import EasyDict

__all__ = ["ConfigLoader"]


class ConfigLoader:
    def __init__(self) -> None:
        self.PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def load_config(config_path: str) -> EasyDict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return EasyDict(config)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    config = config_loader.load_config(
        os.path.join(config_loader.PROJECT_PATH, "configs", "20230221_pcrnet.yaml")
    )
    print(config)
    print(config.train.batch_size)
    print(config["train"]["criterion"])
