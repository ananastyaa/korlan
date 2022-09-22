import os

from image_generate import generate

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_FILE = 'labels.txt'
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'image-data')

if __name__ == '__main__':
    generate(DEFAULT_LABEL_FILE, DEFAULT_FONTS_DIR, DEFAULT_OUTPUT_DIR)
