import tensorflow as tf
from utils import now, load_images_labels, load_features_labels, split_data
from absl import logging
from absl import app
from config import cfg
import os
import datetime
import pickle

FLAGS = dict()
FLAGS['method'] = cfg.METHOD
FLAGS['finetune'] = cfg.FINETUNE


def get_feature_extractor(backbone='vgg'):
    if backbone.lower() == 'vgg':

        vgg = tf.keras.applications.VGG16(include_top=False, weights=cfg.VGG16_WEIGHT,
                                          input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
        vgg.trainable = False
        return vgg

    elif backbone.lower() == 'resnet':
        resnet = tf.keras.applications.ResNet50(include_top=False, weights=cfg.RESNET_WEIGHT,
                                                input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
        resnet.trainable = False
        return resnet
    elif backbone.lower() == 'densnet':
        densnet = tf.keras.applications.DenseNet121(include_top=False, weights=cfg.DENSENET_WEIGHT,
                                                    input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
        densnet.trainable = False
        return densnet
    else:
        return None


def get_transform_model(finetune=False, backbone='vgg'):
    model = tf.keras.models.Sequential()
    if finetune:
        if backbone.lower() == 'vgg':

            vgg = tf.keras.applications.VGG16(include_top=False, weights=cfg.VGG16_WEIGHT,
                                              input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
            vgg.trainable = True
            set_trainable = False
            for layer in vgg.layers:
                if layer.name == 'block5_conv1':
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False
            model.add(vgg)
        elif backbone.lower() == 'resnet':
            resnet = tf.keras.applications.ResNet50(include_top=False, weights=cfg.RESNET_WEIGHT,
                                                    input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
            resnet.trainable = False
            for layer in resnet.layers:
                if layer.name.startswith('conv5'):
                    layer.trainable = True
                else:
                    layer.trainable = False
            model.add(resnet)
        elif backbone.lower() == 'densnet':
            densnet = tf.keras.applications.DenseNet121(include_top=False, weights=cfg.DENSENET_WEIGHT,
                                                        input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
            densnet.trainable = False
            for layer in densnet.layers:
                if layer.name.startswith('conv5'):
                    layer.trainable = True
                else:
                    layer.trainable = False
            model.add(densnet)
        else:
            return None
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    else:
        if backbone.lower() == 'vgg':
            model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 512)))
        elif backbone.lower() == 'resnet':
            model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        else:
            model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1024)))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                    activity_regularizer=tf.keras.regularizers.l1(0.01)))

    return model


def get_none_transform_model(backbone='vgg'):
    model = tf.keras.models.Sequential()
    if backbone.lower() == 'vgg':
        model.add(tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(cfg.HEIGHT, cfg.WIDTH, 3)))
    elif backbone.lower() == 'resnet':
        model.add(
            tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(cfg.HEIGHT, cfg.WIDTH, 3)))
    elif backbone.lower() == 'densnet':
        model.add(
            tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(cfg.HEIGHT, cfg.WIDTH, 3)))
    else:
        return None
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                    activity_regularizer=tf.keras.regularizers.l1(0.01)))

    return model


def train(argv):
    del argv
    if FLAGS['method']:
        logging.info('{} Transform.'.format(now()))
        if FLAGS['finetune']:
            flag = 'transform_finetune'
            features, labels = load_images_labels()
            logging.info('{} images shape: {}'.format(now(), features.shape))
            logging.info('{} labels shape: {}'.format(now(), labels.shape))
        else:
            flag = 'transform'
            features, labels = load_features_labels(backbone=cfg.BACKBONE)
            logging.info('{} features shape: {}'.format(now(), features.shape))
            logging.info('{} labels shape: {}'.format(now(), labels.shape))

        model = get_transform_model(finetune=FLAGS['finetune'], backbone=cfg.BACKBONE)
        logging.info('{} Use {} backbone'.format(now(), cfg.BACKBONE))
        model.summary()
        # split data
        train_x, train_y, val_x, val_y = split_data(features, labels)
    else:
        logging.info('{} No Transform.'.format(now()))
        flag = 'no_transform'
        images, labels = load_images_labels()
        logging.info('{} images shape: {}'.format(now(), images.shape))
        logging.info('{} labels shape: {}'.format(now(), labels.shape))

        model = get_none_transform_model(backbone=cfg.BACKBONE)
        logging.info('{} Use {} backbone'.format(now(), cfg.BACKBONE))
        model.summary()
        # split data
        train_x, train_y, val_x, val_y = split_data(images, labels)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LR, decay=0.00005),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    prefix = datetime.datetime.now().strftime(format='%y%m%d_%H%M%S')
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(cfg.ROOT, 'logs',
                                 '{}_{}_BS{}_EP{}_{}'.format(prefix, cfg.BACKBONE, cfg.BATCH_SIZE, cfg.EPOCHS, flag)),
            update_freq='epoch',
            write_graph=True
        )
    ]
    start = datetime.datetime.now()
    _ = model.fit(x=train_x, y=train_y, epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE,
                  validation_data=[val_x, val_y], callbacks=callbacks)
    if cfg.FINETUNE:
        model.save(os.path.join(cfg.SAVE_DIR,
                                '{}_{}_BS{}_EP{}_{}.h5'.format(prefix, cfg.BACKBONE, cfg.BATCH_SIZE, cfg.EPOCHS, flag)))
    else:
        # 没有使用finetune，也就是说用的是提取过的特征
        backbone = get_feature_extractor(cfg.BACKBONE)
        final_model = tf.keras.models.Sequential()
        final_model.add(backbone)
        final_model.add(model)
        final_model.save(os.path.join(cfg.SAVE_DIR,
                                      '{}_{}_BS{}_EP{}_{}.h5'.format(prefix, cfg.BACKBONE, cfg.BATCH_SIZE, cfg.EPOCHS,
                                                                     flag)))

    cost = datetime.datetime.now() - start
    logging.info('{} Model training cost {} seconds'.format(now(), cost.total_seconds()))


if __name__ == '__main__':
    app.run(train)
