from itertools import combinations
import json
from glob import glob
from os import environ, cpu_count

from tqdm import tqdm
from scipy.spatial.distance import cosine
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd
from cv2 import imread, resize
from joblib import Parallel, delayed

environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAIN_PATH = '/media/ssd/train/'
RESULT_PATH = '/home/arseny/dev/cdiscount-kaggle/data/cosine.json'


def parse(x):
    # this may need changes depending on your data preprocessing flow
    item, num = x.split('/')[-1].split('.')[0].split('_')
    return {'item': item, 'path': x, 'item_num': f'{item}_{num}'}


def main():
    model = ResNet50(weights='imagenet')
    files = glob(f'{TRAIN_PATH}*.jpg')

    files = tqdm(files, total=len(files), desc='preparing data')
    df = pd.DataFrame(Parallel(n_jobs=cpu_count(), backend='threading')(delayed(parse)(f) for f in files))
    with open(RESULT_PATH, 'w') as out:
        for item, pics in tqdm(df.groupby('item'), desc='processing images', total=len(set(df['item']))):
            if len(pics) > 1:
                imgs = np.array([resize(imread(p), (224, 224)) for p in pics.path]).astype('float32')
                preds = model.predict(preprocess_input(imgs))
                combs = combinations(np.arange(len(pics)), 2)
                result = []
                for comb in combs:
                    i, j = comb
                    result.append({'img_a': pics.item_num.values[i],
                                   'img_b': pics.item_num.values[j],
                                   'distance': round(cosine(preds[i], preds[j]), 3)
                                   })

                out.write(json.dumps(result))


if __name__ == '__main__':
    main()
