import tensorflow as tf
import numpy as np
import datetime
import os
from utils import load_features_labels, split_data, now, load_images_labels
from config import cfg
from absl import logging

FINETUNE = cfg.FINETUNE


@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1 * dy

    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs, **kwargs):
        return GradientReversalOperator(inputs)


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


def get_base_block(finetune=False, backbone='vgg'):
    base_block = tf.keras.models.Sequential()
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
            base_block.add(vgg)
        elif backbone.lower() == 'resnet':
            resnet = tf.keras.applications.ResNet50(include_top=False, weights=cfg.RESNET_WEIGHT,
                                                    input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
            resnet.trainable = False
            for layer in resnet.layers:
                if layer.name.startswith('conv5'):
                    layer.trainable = True
                else:
                    layer.trainable = False
            base_block.add(resnet)
        elif backbone.lower() == 'densnet':
            densnet = tf.keras.applications.DenseNet121(include_top=False, weights=cfg.DENSENET_WEIGHT,
                                                        input_shape=(cfg.HEIGHT, cfg.WIDTH, 3))
            densnet.trainable = False
            for layer in densnet.layers:
                if layer.name.startswith('conv5'):
                    layer.trainable = True
                else:
                    layer.trainable = False
            base_block.add(densnet)
        else:
            return None
        base_block.add(tf.keras.layers.GlobalAveragePooling2D())
    else:
        if backbone.lower() == 'vgg':
            base_block.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 512)))
        elif backbone.lower() == 'resnet':
            base_block.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        else:
            base_block.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1024)))

    base_block.add(tf.keras.layers.Dense(128, activation='relu'))
    base_block.add(tf.keras.layers.Dropout(0.3))
    base_block.add(tf.keras.layers.Dense(64, activation='relu'))
    return base_block


def get_classify_block():
    classify_block = tf.keras.models.Sequential()
    # classify_block.add(tf.keras.layers.Dense(64, activation='relu'))
    classify_block.add(tf.keras.layers.Dense(10, activation='softmax'))
    return classify_block


def get_domain_block():
    domain_block = tf.keras.models.Sequential()
    domain_block.add(GradientReversalLayer())
    # domain_block.add(tf.keras.layers.Dense(64, activation='relu'))
    domain_block.add(tf.keras.layers.Dense(2, activation='softmax'))
    return domain_block


def get_models(finetune=False, backbone='vgg'):
    base_block = get_base_block(finetune=finetune, backbone=backbone)
    base_block.summary()
    classify_block = get_classify_block()
    domain_block = get_domain_block()

    classify_model = tf.keras.models.Sequential([
        base_block,
        classify_block
    ])

    domain_model = tf.keras.models.Sequential([
        base_block,
        domain_block
    ])

    return base_block, domain_block, classify_model, domain_model


clf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.00005)
dc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0002)
base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.0002)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

train_lp_loss = tf.keras.metrics.Mean()
train_dc_loss = tf.keras.metrics.Mean()

train_lp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
train_dc_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

test_lp_loss = tf.keras.metrics.Mean()
test_lp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(x_class, y_class, x_domain, classify_model, domain_model, base_block, domain_block):
    domain_labels = np.concatenate([np.zeros(len(x_class)), np.ones(len(x_domain))])
    x_both = tf.concat([x_class, x_domain], axis=0)

    with tf.GradientTape() as tape:
        y_class_pred = classify_model(x_class, training=True)
        lp_loss = loss(y_class, y_class_pred)
    lp_grad = tape.gradient(lp_loss, classify_model.trainable_variables)

    with tf.GradientTape(persistent=True) as tape:
        y_domain_pred = domain_model(x_both, training=True)
        dc_loss = loss(domain_labels, y_domain_pred)
    fe_grad = tape.gradient(dc_loss, base_block.trainable_variables)
    dc_grad = tape.gradient(dc_loss, domain_block.trainable_variables)
    del tape

    clf_optimizer.apply_gradients(zip(lp_grad, classify_model.trainable_variables))
    dc_optimizer.apply_gradients(zip(dc_grad, domain_block.trainable_variables))
    base_optimizer.apply_gradients(zip(fe_grad, base_block.trainable_variables))

    train_lp_loss(lp_loss)
    train_lp_accuracy(y_class, y_class_pred)
    train_dc_loss(dc_loss)
    train_dc_accuracy(domain_labels, y_domain_pred)


@tf.function
def val_step(x, y, classify_model):
    y_pred = classify_model(x, training=False)
    lp_loss = loss(y, y_pred)
    test_lp_loss(lp_loss)
    test_lp_accuracy(y, y_pred)


def main():
    current_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    train_log_dir = os.path.join(cfg.ROOT, 'logs',
                                 '{}_{}_BS{}_EP{}_domain_adap'.format(current_time, cfg.BACKBONE, cfg.BATCH_SIZE,
                                                                      cfg.EPOCHS), 'train')
    test_log_dir = os.path.join(cfg.ROOT, 'logs',
                                '{}_{}_BS{}_EP{}_domain_adap'.format(current_time, cfg.BACKBONE, cfg.BATCH_SIZE,
                                                                     cfg.EPOCHS),
                                'validation')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    if FINETUNE:
        features, labels = load_images_labels()
    else:
        features, labels = load_features_labels(backbone=cfg.BACKBONE)

    logging.info('{} features shape: {}'.format(now(), features.shape))
    logging.info('{} labels shape: {}'.format(now(), labels.shape))
    train_x, train_y, val_x, val_y = split_data(features, labels)
    base_block, domain_block, classify_model, domain_model = get_models(FINETUNE, cfg.BACKBONE)
    classify_model.summary()
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1400).batch(
        cfg.BATCH_SIZE, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

    val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(cfg.BATCH_SIZE, drop_remainder=False)

    for epoch in range(cfg.EPOCHS):
        for x_class, y_class in train_data:
            domain_idx = np.random.choice(range(len(val_x)), size=len(x_class), replace=False)
            x_domain = val_x[domain_idx]
            train_step(x_class, y_class, x_domain, classify_model, domain_model, base_block, domain_block)

        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', train_lp_loss.result(), step=epoch)

            tf.summary.scalar('epoch_accuracy', train_lp_accuracy.result(), step=epoch)

            tf.summary.scalar('epoch_dc_loss', train_dc_loss.result(), step=epoch)
            tf.summary.scalar('epoch_dc_accuracy', train_dc_accuracy.result(), step=epoch)

        for x_val, y_val in val_data:
            val_step(x_val, y_val, classify_model)

        with test_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', test_lp_loss.result(), step=epoch)
            tf.summary.scalar('epoch_accuracy', test_lp_accuracy.result(), step=epoch)

        template = 'Epoch {}, LP Loss: {}, LP Accuracy: {}, ' \
                   'DC Loss: {}, DC Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        logging.info(template.format(
            epoch + 1,
            train_lp_loss.result(),
            train_lp_accuracy.result() * 100,
            train_dc_loss.result(),
            train_dc_accuracy.result() * 100,
            test_lp_loss.result(),
            test_lp_accuracy.result() * 100
        ))

        train_dc_accuracy.reset_states()
        train_dc_loss.reset_states()
        train_lp_loss.reset_states()
        train_lp_accuracy.reset_states()
        test_lp_loss.reset_states()
        test_lp_accuracy.reset_states()

    classify_model.trainable = False
    if cfg.FINETUNE:
        classify_model.save(
            os.path.join(cfg.SAVE_DIR,
                         '{}_{}_BS{}_EP{}_domain_clf.h5'.format(now(), cfg.BACKBONE, cfg.BATCH_SIZE, cfg.EPOCHS)))

    else:
        # 没有使用finetune，也就是说用的是提取过的特征
        backbone = get_feature_extractor(cfg.BACKBONE)
        final_model = tf.keras.models.Sequential()
        final_model.add(backbone)
        final_model.add(classify_model)
        final_model.save(os.path.join(cfg.SAVE_DIR,
                                      '{}_{}_BS{}_EP{}_domain_clf.h5'.format(now(), cfg.BACKBONE, cfg.BATCH_SIZE,
                                                                             cfg.EPOCHS)))


if __name__ == '__main__':
    main()
