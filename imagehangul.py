import glob
import io
import os
import random
import numpy

from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'image-data')
DEFAULT_IMAGE_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'hangul-images')

if not os.path.exists(DEFAULT_IMAGE_DIR):
    os.makedirs(os.path.join(DEFAULT_IMAGE_DIR))


class ImageHangul:

    def __init__(self):
        self.__width = 64
        self.__height = 64
        self.__count = 0
        self.__distortion_count = 1

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def count(self):
        return self.__count

    @property
    def distortion_count(self):
        return self.__distortion_count

    @count.setter
    def count(self, count):
        self.__count = count

    def generate(self, label_file, fonts_dir):
        """
        Метод для генерации изображений с корейскими символами.
        Использует файлы шрифтов, предоставленных в каталоге шрифтов.
        Ожидается, что каталог шрифтов будет заполнен файлами *.ttf (шрифт TrueType).
        Сгенерированные изображения будут сохранены в указанном выходном каталоге.

        :param label_file  : файл с генерируемым текстом
        :param fonts_dir   : директория со шрифтами
        :param output_dir  : папка, куда складывать результат
        """
        with io.open(label_file, 'r', encoding='utf-8') as f:
            labels = f.read().splitlines()

        fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

        for character in labels:
            for font in fonts:
                self.count += 1
                image = Image.new('L', (self.width, self.height), color=0)
                font = ImageFont.truetype(font, 48)
                drawing = ImageDraw.Draw(image)
                w, h = drawing.textsize(character, font=font)
                drawing.text(
                    ((self.width - w) / 2, (self.height - h) / 2),
                    character, fill=255, font=font
                )
                self.save(image, character)

                for i in range(self.distortion_count):
                    self.count += 1
                    arr = numpy.array(image)
                    distorted_array = self.elastic_distort(
                        arr, alpha=random.randint(30, 36),
                        sigma=random.randint(5, 6)
                    )
                    distorted_image = Image.fromarray(distorted_array)
                    self.save(distorted_image, character)

    def save(self, img, character):
        file_string = 'image_{}.jpeg'.format(self.count)
        file_path = os.path.join(DEFAULT_IMAGE_DIR, file_string)

        if self.count == 1:
            labels_csv = io.open(os.path.join(DEFAULT_OUTPUT_DIR, 'labels-map.csv'), 'w',
                                 encoding='utf-8')
        else:
            labels_csv = io.open(os.path.join(DEFAULT_OUTPUT_DIR, 'labels-map.csv'), 'a',
                                 encoding='utf-8')
        labels_csv.write(u'{},{}\n'.format(file_path, character))

        img.save(file_path, 'JPEG')
        labels_csv.close()

    def elastic_distort(self, image, alpha, sigma):
        """
        Функция искажает изображение.
        :param image: искажаемое изображение
        :param alpha: сила искажения
        :param sigma: стандартное отклонение фильтра Гаусса
        """
        random_state = numpy.random.RandomState(None)
        shape = image.shape

        dx = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma, mode="constant"
        ) * alpha
        dy = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma, mode="constant"
        ) * alpha

        x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        indices = numpy.reshape(y + dy, (-1, 1)), numpy.reshape(x + dx, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)
