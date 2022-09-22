import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

DISTORTION_COUNT = 1  # количество изображений одного шрифта, но с искажениями

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def generate(label_file, fonts_dir, output_dir):
    """
    Функция для генерации изображений с корейскими символами.
    Использует файлы шрифтов, предоставленных в каталоге шрифтов.
    Ожидается, что каталог шрифтов будет заполнен файлами *.ttf (шрифт TrueType).
    Сгенерированные изображения будут сохранены в указанном выходном каталоге.

    Планируется переписать класс Image и уже там, с ними работать.
    :param label_file  : файл с генерируемым текстом
    :param fonts_dir   : директория со шрифтами
    :param output_dir  : папка, куда складывать результат
    """
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
                         encoding='utf-8')

    total_count = 0
    for character in labels:
        for font in fonts:
            total_count += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
            font = ImageFont.truetype(font, 48)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(
                ((IMAGE_WIDTH - w) / 2, (IMAGE_HEIGHT - h) / 2),
                character,
                fill=(255),
                font=font
            )
            file_string = 'image_{}.jpeg'.format(total_count)
            file_path = os.path.join(image_dir, file_string)
            image.save(file_path, 'JPEG')
            labels_csv.write(u'{},{}\n'.format(file_path, character))

            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = 'image_{}.jpeg'.format(total_count)
                file_path = os.path.join(image_dir, file_string)
                arr = numpy.array(image)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                distorted_image = Image.fromarray(distorted_array)
                distorted_image.save(file_path, 'JPEG')
                labels_csv.write(u'{},{}\n'.format(file_path, character))

    print('Сгенерировано {} изображений.'.format(total_count))
    labels_csv.close()


def elastic_distort(image, alpha, sigma):
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
