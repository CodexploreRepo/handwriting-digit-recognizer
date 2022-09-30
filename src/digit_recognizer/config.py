"""
This is the config file of Digit Recognizer
"""
import os
import pathlib

SEED = 42

# --- PATH ----
MAIN_PATH = pathlib.Path(__file__).resolve().parents[2]
# handwriting-digit-recognizer/data
DATA_PATH = MAIN_PATH / "data"
MODEL_PATH = MAIN_PATH / "model_chkpt"
RESULT_PATH = MAIN_PATH / "docs" / "submissions"
# --- CPU ---
NUM_WORKERS = os.cpu_count()
