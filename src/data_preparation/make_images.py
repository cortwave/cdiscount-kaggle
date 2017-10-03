import bson
import io
import os
from skimage.io import imsave
from skimage.data import imread
from tqdm import tqdm
import logging
import warnings

logger = logging.getLogger("ImagesTransfer")


def make_images(input_file, output_dir):
    data = bson.decode_file_iter(open(input_file, 'rb'))
    with warnings.catch_warnings():
        for item in tqdm(data):
            product_id = item['_id']
            for ix, pic in enumerate(item['imgs']):
                warnings.filterwarnings('error')
                try:
                    img = imread(io.BytesIO(pic['picture']))
                    imsave(f"{output_dir}/{product_id}_{ix}.jpg", img)
                except Warning:
                    warnings.filterwarnings('ignore')
                    img = imread(io.BytesIO(pic['picture']))
                    imsave(f"../../data/images/low_contrast/{product_id}_{ix}.jpg", img)
                    imsave(f"{output_dir}/{product_id}_{ix}.jpg", img)


def make_train():
    logger.info("Train set transforming")
    out_path = "../../data/images/train"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    make_images("../../data/train.bson", out_path)


def make_test():
    logger.info("Test set transforming")
    out_path = "../../data/images/test"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    make_images("../../data/test.bson", out_path)


if __name__ == '__main__':
    path = "../../data/images"
    if not os.path.exists(path):
        os.mkdir(path)
    make_train()
    make_test()