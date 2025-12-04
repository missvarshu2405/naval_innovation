import yaml

class Config:
    def __init__(self, cfg_path="config/model_config.yaml"):
        with open(cfg_path, 'r') as file:
            self.cfg = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.cfg.get(key, default)

CONFIG = Config()
