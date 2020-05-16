import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

SAVED_DIR = os.path.join(CURRENT_DIR, 'static', 'saved')
DEFAULT_IMAGE_PATH = os.path.join(SAVED_DIR, 'default.jpg')
MODEL_NAME = 'model.h5'
MODEL_PATH = os.path.join(CURRENT_DIR, 'model', MODEL_NAME)
HEIGHT, WIDTH = 224, 224

LABEL_MAP = {
    0: 'Bus',
    1: 'Family Sedan',
    2: 'Fire Engine',
    3: 'Heavy Truck',
    4: 'Jeep',
    5: 'Minibus',
    6: 'Racing Car',
    7: 'SUV',
    8: 'Taxi',
    9: 'Truck'
}
