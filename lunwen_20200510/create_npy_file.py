import numpy as np
import tensorflow as tf
from config import cfg
import os
from absl import logging
from utils import now
import tqdm
import time

logging.set_verbosity(logging.INFO)


def parse_line(line):
    tmp = tf.strings.split(line)
    filename, label = tmp[0], tmp[1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [cfg.HEIGHT, cfg.WIDTH])
    label = tf.strings.to_number(label, tf.int32)
    return image, label


def create():
    """

    :return:
    """
    start = time.time()
    if not os.path.exists(os.path.join(cfg.ROOT, 'annotation.txt')):
        logging.error('{} annotation.txt not exists.'.format(now()))
        return None

    image_dataset = tf.data.TextLineDataset(os.path.join(cfg.ROOT, 'annotation.txt')).map(parse_line)
    images = []
    labels = []
    for image, label in tqdm.tqdm(image_dataset):
        images.append(image.numpy())
        labels.append(label.numpy())

    images = np.array(images)
    labels = np.array(labels)

    np.save(os.path.join(cfg.NPY_DIR, 'images'), images)
    np.save(os.path.join(cfg.NPY_DIR, 'labels'), labels)

    logging.info('{} Total: {}'.format(now(), len(images)))
    logging.info('{} Images shape: {}'.format(now(), images.shape))
    logging.info('{} Labels shape: {}'.format(now(), labels.shape))

    if cfg.BACKBONE == 'vgg':
        logging.info('{} Load pre-trained {} weights {}'.format(now(), cfg.BACKBONE, cfg.VGG16_WEIGHT))

        backbone = tf.keras.applications.VGG16(
            include_top=False,
            weights=cfg.VGG16_WEIGHT,
            input_shape=(cfg.HEIGHT, cfg.WIDTH, 3)
        )
    elif cfg.BACKBONE == 'resnet':
        logging.info('{} Load pre-trained {} weights {}'.format(now(), cfg.BACKBONE, cfg.RESNET_WEIGHT))

        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=cfg.RESNET_WEIGHT,
            input_shape=(cfg.HEIGHT, cfg.WIDTH, 3)
        )
    else:
        logging.info('{} Load pre-trained {} weights {}'.format(now(), cfg.BACKBONE, cfg.DENSENET_WEIGHT))

        backbone = tf.keras.applications.DenseNet121(
            include_top=False,
            weights=cfg.DENSENET_WEIGHT,
            input_shape=(cfg.HEIGHT, cfg.WIDTH, 3)
        )

    backbone.trainable = False
    print(backbone.summary())

    logging.info('{} Start extract feature.'.format(now()))
    features = []
    for idx in tqdm.tqdm(range(len(images))):
        feature = backbone.predict(images[idx: idx + 1])
        features.append(feature)

    features = np.concatenate(features)

    logging.info('{} Features shape: {}'.format(now(), features.shape))

    np.save(os.path.join(cfg.NPY_DIR, '{}_features'.format(cfg.BACKBONE)), features)
    np.save(os.path.join(cfg.NPY_DIR, '{}_labels'.format(cfg.BACKBONE)), labels)

    logging.info('{} Complete, cost: {}'.format(now(), time.time() - start))


if __name__ == '__main__':
    create()
