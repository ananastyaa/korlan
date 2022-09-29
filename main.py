import os

from imagehangul import ImageHangul
from TFRecordsConverter import TFRecordsConverter

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_FILE = 'labels.txt'
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'fonts')
DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, 'image-data/labels-map.csv')
DEFAULT_OUTPUT_TF_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')

if __name__ == '__main__':
    imagehangul = ImageHangul()
    imagehangul.generate(DEFAULT_LABEL_FILE, DEFAULT_FONTS_DIR)
    print('Сгенерировано {} изображений.'.format(imagehangul.count))
    converter = TFRecordsConverter(DEFAULT_LABEL_CSV,
                                   DEFAULT_LABEL_FILE, 2, 1)
    converter.convert()
