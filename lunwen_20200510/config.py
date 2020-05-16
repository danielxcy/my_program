import os
import easydict as ed

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

cfg = ed.EasyDict()

cfg.ROOT = CURRENT_DIR
cfg.HEIGHT = 224
cfg.WIDTH = 224
cfg.NPY_DIR = os.path.join(CURRENT_DIR, 'npy_files')
cfg.IMAGE_DIR = os.path.join(CURRENT_DIR, 'images')
cfg.SAVE_DIR = os.path.join(CURRENT_DIR, 'models')

cfg.VGG16_WEIGHT = os.path.join(CURRENT_DIR, 'pre_trained_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
cfg.RESNET_WEIGHT = os.path.join(CURRENT_DIR,
                                 'pre_trained_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
cfg.DENSENET_WEIGHT = os.path.join(CURRENT_DIR,
                                   'pre_trained_weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')


cfg.BACKBONE = 'vgg'  # vgg 或者 resnet 或者 densnet
cfg.EPOCHS = 1
cfg.BATCH_SIZE = 10
cfg.LR = 0.0001
cfg.FINETUNE = True  # 是否启用finetune
cfg.METHOD = True  # 是否 transform
