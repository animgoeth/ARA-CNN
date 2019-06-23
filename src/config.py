from keras.optimizers import Adam

# General image options
IMAGE_SIZE = (128, 128)  # input images are scaled to this resolution
COLOR_TYPE = 'rgb'  # change to grayscale if your images are monochrome
CHANNELS = 1 if COLOR_TYPE == 'grayscale' else 3

# Main class dictionary - this needs to match folder names in your training dataset
CLASS_DICT = {
    "01_TUMOR": 0,
    "02_STROMA": 1,
    "03_COMPLEX": 2,
    "04_LYMPHO": 3,
    "05_DEBRIS": 4,
    "06_MUCOSA": 5,
    "07_ADIPOSE": 6,
    "08_EMPTY": 7,
}

# Default optimizer
DEFAULT_OPTIMIZER = Adam()

# Cliping EPS
EPS = 1e-6
