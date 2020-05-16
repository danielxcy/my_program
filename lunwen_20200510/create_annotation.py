import os
import datetime
from config import cfg
from absl import logging
import tqdm
import time
from utils import now

logging.set_verbosity(logging.INFO)


def create_annotation():
    """
    创建annotation.txt文件
    :return:
    """
    start = time.time()
    label_map = {
        'bus': 0,
        'family_sedan': 1,
        'fire_engine': 2,
        'heavy_truck': 3,
        'jeep': 4,
        'minibus': 5,
        'racing_car': 6,
        'SUV': 7,
        'taxi': 8,
        'truck': 9
    }
    logging.info('{} Create annotation.txt'.format(now()))
    with open('annotation.txt', 'w', encoding='utf8') as f:
        for classes in os.listdir(cfg.IMAGE_DIR):
            logging.info('{} Class: {}'.format(now(), classes))
            pbr = tqdm.tqdm(os.listdir(os.path.join(cfg.IMAGE_DIR, classes)))
            try:
                for file_name in pbr:
                    f.write(
                        "{} {}\n".format(os.path.join(cfg.IMAGE_DIR, classes, file_name),
                                         label_map[classes]))
            except Exception as e:
                logging.error('{} {}'.format(now(), e))
                pbr.close()

    logging.info('{} Create annotation.txt completed, cost {}'.format(now(), time.time() - start))


if __name__ == '__main__':
    create_annotation()
