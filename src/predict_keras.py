import logging
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
from cv2 import imread, resize
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from fire import Fire
from imgaug import augmenters as iaa

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

TTA_ROUNDS = 0


def main(model_name):
    batch_size = 256

    labels_map = pd.read_csv("../data/labels_map.csv")
    labels_map = {k: v for k, v in zip(labels_map.label_id, labels_map.category_id)}

    test_files = glob('../data/images/test/*_0.jpg')
    models = glob(f'../results/{model_name}*.h5')

    cropper = iaa.Crop(px=((0, 20), (0, 20), (0, 20), (0, 20)), keep_size=True)
    batches = np.array_split(test_files, (len(test_files) // batch_size) + 1)

    for model_path in models:
        result = []

        model = load_model(model_path, compile=False)
        for batch in tqdm(batches, desc=f'Batches processed for {model_path}'):
            images = np.array([resize(imread(img_path), (150, 150)) for img_path in batch])
            ids = [x.split('/')[-1].split('_')[0] for x in batch]

            if TTA_ROUNDS:
                for _ in range(TTA_ROUNDS):
                    images = cropper.augment_images(images).astype('float32')
                    images = preprocess_input(images)

                    labels = model.predict(images).argmax(axis=-1)
                    labels = (labels_map[x] for x in labels)

                    for label, id_ in zip(labels, ids):
                        result.append({'_id': id_, 'category_id': label})
            else:
                images = cropper.augment_images(images).astype('float32')
                images = preprocess_input(images)

                labels = model.predict(images).argmax(axis=-1)
                labels = (labels_map[x] for x in labels)

                for label, id_ in zip(labels, ids):
                    result.append({'_id': id_, 'category_id': label})

        pd.DataFrame(result).to_csv(f'../results/submit_{model_path.split("/")[-1].split(".")[0]}.csv.gz',
                                    index=False, compression='gzip')
        # pd.DataFrame(result).to_csv(f'../results/submit_{model_name}.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    Fire(main)
