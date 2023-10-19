from os.path import join, dirname

MODELS_DIR = join(dirname(dirname(dirname(__file__))), "models")

IMAGE_MODELS_DIR = join(MODELS_DIR, "image")
TEXT_MODELS_DIR = join(MODELS_DIR, "text")

FACE_CLASSIFIER = join(IMAGE_MODELS_DIR, "lbpcascade_frontalface_improved.xml")

SENTIMENT_API_RESULTS = join(TEXT_MODELS_DIR, "sentiments.data")
SENTILEX_DATA_PT = join(TEXT_MODELS_DIR, "SentiLex-flex-PT02.txt")
