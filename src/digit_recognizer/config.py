"""
This is the config file of Digit Recognizer
"""
import pathlib

SEED = 42

# handwriting-digit-recognizer/data
DATA_PATH = pathlib.Path(__file__).resolve().parents[2] / "data"
