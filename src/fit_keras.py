import logging
from os import environ
from collections import Counter

import numpy as np
import pandas as pd
from cv2 import imread, resize
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_xcept
from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Nadam
from keras.layers import Dense, Concatenate, Input
from keras.utils import to_categorical
from fire import Fire
from imgaug import augmenters as iaa

from keras_utils import threadsafe_generator, AdamAccum

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

        self.counter = self.count_imgs_per_item(df)
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
        self.augmenter = iaa.Sequential([iaa.Fliplr(p=.25),
                                         iaa.Flipud(p=.25),
                                         # iaa.Crop(px=(10, 10, 10, 10), keep_size=False)
                                         # iaa.GaussianBlur(sigma=(.05, .3))
                                         ],
                                        random_order=False)

    @staticmethod
    def count_imgs_per_item(df):
        images = [x.split('_')[0] for x in df.image_id.values]
        return Counter(images)

    def _load(self, image):
        img = imread(f"/media/ssd/train/{image}.jpg")
        # img = imread(f"../data/images/train/{image}.jpg")
        cnt = self.counter[image.split('_')[0]]
        if self.shape != (180, 180):
            img = resize(img, self.shape)
        return img, cnt

    @threadsafe_generator
    def get_batch(self, batch_size):
        batch_size = int(batch_size)
        while True:
            idx = np.random.randint(0, self.num_images, batch_size)
            yield self._get_images(idx)

    def _get_images(self, idx_batch):
        x_img, x_cnt = zip(*[self._load(self.images[idx]) for idx in idx_batch])
        x_img = np.array(x_img).astype('float32')
        x_cnt = np.array(x_cnt)

        if self.aug:
            x_img = self.augmenter.augment_images(x_img)
        if self.transform:
            x_img = self.transform(x_img)
        y = to_categorical(self.labels[idx_batch],
                           num_classes=self.num_classes)
        return [x_img, x_cnt], y


def get_callbacks(model_name, fold):
    model_checkpoint = ModelCheckpoint(f'../results/{model_name}_{fold}.h5',
                                       monitor='val_loss',
                                       save_best_only=True, verbose=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=6)
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
    cnt_input = Input(shape=(1,))
    cnt = Dense(1, activation='relu')(cnt_input)
    x = Concatenate()([cnt, x])
    x = Dense(n_classes, activation='relu')(x)
    predictions = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=[base_model.input, cnt_input], outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Nadam(clipvalue=3, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])
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

    fname = f'../results/{model_name}_{n_fold}.h5'
    frozen_epochs = 1

    try:
        model = load_model(fname)
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

    model.compile(optimizer=AdamAccum(clipvalue=4), loss='categorical_crossentropy', metrics=['accuracy'])
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
