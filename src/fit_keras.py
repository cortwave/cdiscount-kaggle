import logging
from threading import Lock
from os import environ

import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.misc import imresize
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_xcept
from keras.applications.xception import Xception
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils import to_categorical

from fire import Fire
from imgaug import augmenters as iaa

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)


class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class Dataset:
    def __init__(self, n_fold, n_folds, shape, transform=None, train=True, aug=False):
        if train:
            folds = list(range(n_folds))
            folds.remove(n_fold)
            train_dfs = [pd.read_csv(f"../data/fold_{i}.csv") for i in folds]
            df = pd.concat(train_dfs)
        else:
            df = pd.read_csv(f"../data/fold_{n_fold}.csv")
        labels_map = pd.read_csv("../data/labels_map.csv")
        labels_map.index = labels_map.category_id
        self.num_classes = labels_map.shape[0]
        self.labels_map = labels_map
        self.images = df.image_id.values
        self.labels = df.category_id.values
        self.transform = transform
        self.shape = shape
        self.aug = aug
        self.augmenter = iaa.Sequential([iaa.Fliplr(p=.25),
                                         iaa.Flipud(p=.25),
                                         # iaa.GaussianBlur(sigma=(.05, .3))
                                         ],
                                        random_order=False)

    def _load(self, image):
        img = imread(f"../data/images/train/{image}.jpg")
        if self.shape != (180, 180):
            img = imresize(img, self.shape)
        return img

    @threadsafe_generator
    def get_batch(self, batch_size):
        while True:
            idx = np.random.choice(np.arange(self.images.shape[0]), int(batch_size), replace=False)
            yield self._get_images(idx)

    def _get_images(self, idx_batch):
        X = np.array([self._load(self.images[idx]) for idx in idx_batch]).astype('float32')

        if self.aug:
            X = self.augmenter.augment_images(X)
        if self.transform:
            X = self.transform(X)
        y = to_categorical(np.array([self.labels_map.ix[self.labels[idx]].label_id for idx in idx_batch]),
                           num_classes=self.num_classes)
        return X, y


def get_callbacks(model_name, fold):
    model_checkpoint = ModelCheckpoint(f'../results/{model_name}_{fold}.h5',
                                       monitor='val_loss',
                                       save_best_only=True, verbose=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=2)
    return [model_checkpoint, es, reducer]


def get_model(model_name, n_classes):
    if model_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(197, 197, 3), pooling='avg')
        preprocess = preprocess_input
        shape = (197, 197)
    elif model_name == 'inception':
        base_model = InceptionV3(include_top=False, input_shape=(180, 180, 3), pooling='avg')
        preprocess = preprocess_xcept
        shape = (180, 180)
    elif model_name == 'xception':
        base_model = Xception(include_top=False, input_shape=(180, 180, 3), pooling='avg')
        preprocess = preprocess_xcept
        shape = (180, 180)
    else:
        raise ValueError('Network name is undefined')

    x = base_model.output
    predictions = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(clipvalue=2), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, preprocess, shape


def fit_model(model_name, batch_size=64, n_fold=0, cuda='1'):
    environ['CUDA_VISIBLE_DEVICES'] = str(cuda)

    train = Dataset(n_fold=n_fold,
                    n_folds=5,
                    transform=None,
                    train=True,
                    shape=None,
                    aug=False)

    model, preprocess, shape = get_model(model_name, train.num_classes)

    train = Dataset(n_fold=n_fold,
                    n_folds=5,
                    transform=preprocess,
                    train=True,
                    shape=shape,
                    aug=False)

    val = Dataset(n_fold=0,
                  n_folds=5,
                  transform=preprocess,
                  train=False,
                  shape=shape,
                  aug=False)

    model.fit_generator(train.get_batch(batch_size),
                        epochs=500,
                        steps_per_epoch=1000,
                        validation_data=val.get_batch(batch_size),
                        workers=8,
                        validation_steps=100,
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')]
                        )

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(clipvalue=3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train.get_batch(batch_size),
                        epochs=500,
                        steps_per_epoch=500,
                        validation_data=val.get_batch(batch_size),
                        workers=8,
                        validation_steps=100,
                        callbacks=get_callbacks(model_name, fold)
                        )


if __name__ == '__main__':
    Fire(fit_model)
