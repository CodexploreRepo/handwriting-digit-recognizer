"""
This is the config file of Digit Recognizer
"""
import pathlib

SEED = 42

# handwriting-digit-recognizer/data
MAIN_PATH = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = MAIN_PATH / "data"
MODEL_PATH = MAIN_PATH / "model_chkpt"
