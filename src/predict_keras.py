import logging
import json
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
from cv2 import imread
from keras.applications.xception import preprocess_input as preprocess_xcept
from keras.models import load_model
from fire import Fire

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)


def crop(img, shape, option):
    margin = 180 - shape
    half = int(margin / 2)
    crops = [lambda x: x[:-margin, :-margin, ...],
             lambda x: x[:-margin, margin:, ...],
             lambda x: x[margin:, margin:, ...],
             lambda x: x[margin:, :-margin, ...],
             lambda x: x[half:-half, half:-half, ...],
             ]

    return crops[option](img)


def load_img(img_path, crop_option, shape=150):
    img = imread(img_path)

    img = crop(img, shape, crop_option)
    if np.random.rand() > .7:
        img = np.fliplr(img)

    return img


def main(model_name, use_tail=False):
    batch_size = 256

    labels_map = pd.read_csv("../data/labels_map.csv")
    labels_map = {k: v for k, v in zip(labels_map.label_id, labels_map.category_id)}

    if use_tail:
        test_files = list(set(glob('../data/images/test/*.jpg')) - set(glob('../data/images/test/*_0.jpg')))
    else:
        test_files = glob('../data/images/test/*_0.jpg')

    models = glob(f'../results/{model_name}*.h5')

    batches = np.array_split(test_files, (len(test_files) // batch_size) + 1)

    for model_path in models:
        model = load_model(model_path, compile=False)

        fname = f'../results/submit_{model_path.split("/")[-1].split(".")[0]}.json'
        if use_tail:
            fname = fname.replace('.json', '_tail.json')

        with open(fname, 'w') as out:
            for batch in tqdm(batches, desc=f'Batches processed for {model_path}'):
                ids = [x.split('/')[-1].split('_')[0] for x in batch]
                tta_proba = []

                for crop_option in range(0, 5):
                    images = np.array([load_img(img_path, crop_option) for img_path in batch]).astype('float32')
                    images = preprocess_xcept(images)

                    labels = model.predict(images)
                    tta_proba.append(labels)

                labels = np.round(np.mean(tta_proba, axis=0), 2)

                for label, id_ in zip(labels, ids):
                    d = {'_id': id_}
                    for i, proba in enumerate(label):
                        label_id = labels_map[i]
                        if proba > .01:
                            d[str(label_id)] = int(proba * 100)
                    out.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    Fire(main)
