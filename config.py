import os
from types import SimpleNamespace

CONFIG = SimpleNamespace()
CONFIG.DATA_PATH = os.path.join("./data")
CONFIG.STORE_TRUE = None
CONFIG.DIV2K_TEST = os.path.join(CONFIG.DATA_PATH, "div2k", "test_data")
CONFIG.DIV2K_TRAIN = os.path.join(CONFIG.DATA_PATH, "div2k", "train_data")
CONFIG.KODAK = os.path.join(CONFIG.DATA_PATH, "Kodak")