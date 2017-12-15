import logging

import numpy as np
import pandas as pd
from cv2 import imread
from keras.applications.xception import Xception, preprocess_input as preprocess_xcept
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Nadam, SGD
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from fire import Fire
from keras_utils import threadsafe_generator

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, n_fold, n_folds, shape, transform=None, train=True, aug=False):
        if train:
            folds = list(range(n_folds))
            folds.remove(n_fold)
            train_dfs = [pd.read_csv(f"../data/fold_{i}.csv") for i in folds]
            df = pd.concat(train_dfs)
        else:
            df = pd.read_csv(f"../data/fold_{n_fold}.csv")

        df = df[df.image_id.str.endswith('_0')]
        labels_map = pd.read_csv("../data/labels_map.csv")
        labels_map = {k: v for k, v in zip(labels_map.category_id, labels_map.label_id)}
        self.num_classes = len(labels_map)
        self.images = df.image_id.values
        self.labels = np.array([labels_map[x] for x in df.category_id.values])
        self.num_images = self.images.shape[0]
        self.transform = transform
        self.shape = shape
        self.aug = aug


    @staticmethod
    def _crop(img, shape, option):
        margin = 180 - shape
        half = int(margin / 2)
        crops = [lambda x: x[:-margin, :-margin, ...],
                 lambda x: x[:-margin, margin:, ...],
                 lambda x: x[margin:, margin:, ...],
                 lambda x: x[margin:, :-margin, ...],
                 lambda x: x[half:-half, half:-half, ...],
                 ]

        return crops[option](img)

    def _load(self, image):
        img = imread(f"/media/ssd/train/{image}.jpg")
        img = self._crop(img, self.shape, np.random.randint(0, 5))
        if np.random.rand() > .7:
            img = np.fliplr(img)

        return img

    @threadsafe_generator
    def get_batch(self, batch_size):
        batch_size = int(batch_size)
        while True:
            idx = np.random.randint(0, self.num_images, batch_size)
            yield self._get_images(idx)

    def _get_images(self, idx_batch):
        X = np.array([self._load(self.images[idx]) for idx in idx_batch]).astype('float32')

        if self.transform:
            X = self.transform(X)
        y = to_categorical(self.labels[idx_batch],
                           num_classes=self.num_classes)
        return X, y


def get_callbacks(model_name, fold):
    model_checkpoint = ModelCheckpoint(f'../results/{model_name}_{fold}.h5',
                                       monitor='val_loss',
                                       save_best_only=True, verbose=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=6)
    return [model_checkpoint, es, reducer]


def get_model(model_name, n_classes):
    if model_name == 'xception':
        shape = 150
        base_model = Xception(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        preprocess = preprocess_xcept
        drop = .1
    elif model_name == 'incres':
        shape = 150
        base_model = InceptionResNetV2(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        preprocess = preprocess_input
        drop = .4
    else:
        raise ValueError('Network name is undefined')

    x = base_model.output
    x = Dropout(drop)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Nadam(clipvalue=3, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, preprocess, shape


def fit_model(model_name, batch_size=64, n_fold=0):

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

    fname = f'../results/{model_name}_{n_fold}.h5'
    frozen_epochs = 1

    try:
        model = load_model(fname, compile=False)
    except OSError:
        model.fit_generator(train.get_batch(batch_size),
                            epochs=frozen_epochs,
                            steps_per_epoch=1000,
                            validation_data=val.get_batch(batch_size),
                            workers=8,
                            validation_steps=100,
                            use_multiprocessing=False,
                            max_queue_size=50,
                            callbacks=[ModelCheckpoint(f'../results/{model_name}_{n_fold}.h5',
                                                       monitor='val_acc',
                                                       save_best_only=True, verbose=0)]
                            )

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=SGD(clipvalue=4, momentum=.9, nesterov=True), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(train.get_batch(batch_size),
                        epochs=500,
                        steps_per_epoch=2000,
                        validation_data=val.get_batch(batch_size),
                        workers=8,
                        use_multiprocessing=False,
                        validation_steps=100,
                        callbacks=get_callbacks(model_name, n_fold),
                        initial_epoch=frozen_epochs,
                        )


if __name__ == '__main__':
    Fire(fit_model)
