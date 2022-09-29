import os

from image_generate import generate
from imagehangul import ImageHangul

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_FILE = 'labels.txt'
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'fonts')

if __name__ == '__main__':
    imagehangul = ImageHangul()
    imagehangul.generate(DEFAULT_LABEL_FILE, DEFAULT_FONTS_DIR)
    print('Сгенерировано {} изображений.'.format(imagehangul.count))