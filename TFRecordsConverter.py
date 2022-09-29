from __future__ import division

import io
import math
import os
import random

import numpy as np
import tensorflow as tf

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_TF_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')

if not os.path.exists(DEFAULT_OUTPUT_TF_DIR):
    os.makedirs(os.path.join(DEFAULT_OUTPUT_TF_DIR))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsConverter:
    """Класс конвертирует изображения в TFRecords."""

    def __init__(self, labels_csv, label_file,
                 num_train_tfr, num_test_tfr):

        self.num_train_tfr = num_train_tfr
        self.num_test_tfr = num_test_tfr

        self.images, self.labels = self.process_image_labels(labels_csv, label_file)
        self.counter = 0

    def process_image_labels(self, labels_csv, label_file):
        """ Создаем два перетасованных списка изображений и меток. """
        labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
        labels_file = io.open(label_file, 'r', encoding='utf-8').read().splitlines()

        # индексируем все изначальные метки (которые рисовали)
        label_dict = dict(enumerate(labels_file))
        label_dict = dict(zip(label_dict.values(), label_dict.keys()))

        # разделяем метки от изображений
        images, labels = zip(*[row.strip().split(',') for row in labels_csv])
        labels = [label_dict[label] for label in labels]

        # перемешиваем метки и изображения, чтобы не шли по порядку
        shuffled_indices = list(range(len(images)))
        random.shuffle(shuffled_indices)
        images = [images[i] for i in shuffled_indices]
        labels = [labels[i] for i in shuffled_indices]
        return images, labels

    def write_tfrecords_file(self, output_path, indices):
        """
        Записываем TFRecords в файл
        Примечание: Example - это формат данных, содержащий хранилище ключ-значение.
        Каждый ключ соответствует функциональному сообщению.
        В этом случае он содержит две функции. Одним из них будет список
        байтов для необработанного изображения data, а другой будет Int64List,
        содержащим индекс соответствующей метки в списке меток из файла."""
        writer = tf.io.TFRecordWriter(output_path)
        for i in indices:
            filename = self.images[i]
            label = self.labels[i]
            with tf.io.gfile.GFile(filename, 'rb') as f:
                im_data = f.read()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/class/label': _int64_feature(label),
                'image/encoded': _bytes_feature(tf.compat.as_bytes(im_data))}))
            writer.write(example.SerializeToString())
            self.counter += 1
        writer.close()

    def convert(self):
        """
        Метод преобразовывает изображения в TFRecords.
        Разделяем данные на обучающий и тестовый наборы, затем
        делим каждый набор данных на указанное количество сегментов TFRecords.
        """
        all_images = len(self.images)
        # 15% изображений в тестовую выборку, ост обучающая
        num_test_image = int(all_images * .15)
        num_train_image = all_images - num_test_image
        image_one_file = int(math.ceil(num_train_image / self.num_train_tfr))

        start = 0
        for i in range(1, self.num_train_tfr):
            path_tfr = os.path.join(DEFAULT_OUTPUT_TF_DIR, 'train-{}.tfrecords'.format(str(i)))
            # получить индексы для изображений и меток текущего TFR-файла
            file_indices = np.arange(start, start + image_one_file, dtype=int)
            start = start + image_one_file
            self.write_tfrecords_file(path_tfr, file_indices)

        # оставшиеся изображения войдут в последний файл
        file_indices = np.arange(start, num_train_image, dtype=int)
        final_path_tfr = os.path.join(DEFAULT_OUTPUT_TF_DIR,
                                      'train-{}.tfrecords'.format(str(self.num_train_tfr)))
        self.write_tfrecords_file(final_path_tfr, file_indices)

        image_one_file = math.ceil(num_test_image / self.num_test_tfr)
        start = num_train_image
        for i in range(1, self.num_test_tfr):
            path_tfr = os.path.join(DEFAULT_OUTPUT_TF_DIR,
                                    'test-{}.tfrecords'.format(str(i)))
            file_indices = np.arange(start, start + image_one_file, dtype=int)
            start = start + image_one_file
            self.write_tfrecords_file(path_tfr, file_indices)

        file_indices = np.arange(start, all_images, dtype=int)
        final_path_tfr = os.path.join(DEFAULT_OUTPUT_TF_DIR,
                                      'test-{}.tfrecords'.format(
                                          str(self.num_test_tfr)))
        self.write_tfrecords_file(final_path_tfr, file_indices)
