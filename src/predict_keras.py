import logging
from os import environ
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
from cv2 import imread, resize
from keras.applications.xception import preprocess_input
from keras.models import load_model
from fire import Fire
from imgaug import augmenters as iaa

from keras_utils import AdamAccum

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)


def read_image(img):
    return resize(imread(img), (197, 197))


def main(model_name, cuda=''):
    environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
    batch_size = 256

    labels_map = pd.read_csv("../data/labels_map.csv")
    labels_map = {k: v for k, v in zip(labels_map.label_id, labels_map.category_id)}

    test_files = glob('../data/images/test/*_0.jpg')
    fname = f'../results/{model_name}.h5'
    model = load_model(fname, {'AdamAccum':AdamAccum})

    cropper = iaa.Crop(px=(10, 10, 10, 10))
    batches = np.array_split(test_files, (len(test_files) // batch_size) + 1)

    result = []
    for batch in tqdm(batches, desc='Batches processed'):
        images = np.array([read_image(img_path) for img_path in batch]).astype('float32')
        # images = cropper.augment_images(images)
        images = preprocess_input(images)
        labels = model.predict(images).argmax(axis=-1)
        labels = (labels_map[x] for x in labels)
        ids = (x.split('/')[-1].split('_')[0] for x in batch)

        for label, id_ in zip(labels, ids):
            result.append({'_id': id_, 'category_id': label})

    pd.DataFrame(result).to_csv(f'../results/submit_{model_name}.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    Fire(main)
