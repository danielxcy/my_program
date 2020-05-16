import datetime
import os
from config import cfg
from absl import logging
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

logging.set_verbosity(logging.INFO)


def now():
    return datetime.datetime.now().strftime(format='%y%m%d_%H%M%S')


def load_features_labels(backbone):
    logging.info(
        '{} Load {}'.format(now(), os.path.join(cfg.NPY_DIR, '{}_features.npy'.format(backbone))))
    logging.info(
        '{} Load {}'.format(now(), os.path.join(cfg.NPY_DIR, '{}_labels.npy'.format(backbone))))
    features = np.load(os.path.join(cfg.NPY_DIR, '{}_features.npy'.format(backbone)))
    labels = np.load(os.path.join(cfg.NPY_DIR, '{}_labels.npy'.format(backbone)))
    return features, labels


def load_images_labels():
    logging.info(
        '{} Load {}'.format(now(), os.path.join(cfg.NPY_DIR, 'images.npy')))
    logging.info(
        '{} Load {}'.format(now(), os.path.join(cfg.NPY_DIR, 'labels.npy')))
    images = np.load(os.path.join(cfg.NPY_DIR, 'images.npy'))
    labels = np.load(os.path.join(cfg.NPY_DIR, 'labels.npy'))
    return images, labels


def split_data(features, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=137)
    train_x, train_y, val_x, val_y = [None] * 4

    for train_idx, val_idx in sss.split(features, labels):
        train_x = features[train_idx]
        val_x = features[val_idx]

        train_y = labels[train_idx]
        val_y = labels[val_idx]
    logging.info('{} train_x shape: {}'.format(now(), train_x.shape))
    logging.info('{} val_x shape: {}'.format(now(), val_x.shape))
    logging.info('{} train_y shape: {}'.format(now(), train_y.shape))
    logging.info('{} val_y shape: {}'.format(now(), val_y.shape))
    return train_x, train_y, val_x, val_y



