import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_PATH, "/models")
print(MODEL_PATH)

LABELS = {
    0: "CCAT",
    1: "ECAT",
    2: "GCAT",
    3: "MCAT"
}

MAX_LEN = 128 # max sequences length


