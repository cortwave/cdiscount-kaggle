import logging

import numpy as np
import pandas as pd
from cv2 import imread, resize
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_xcept
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Nadam, SGD
from keras.layers import Dense, Dropout
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
        self.augmenter = iaa.Sequential([iaa.Fliplr(p=.3),
                                         iaa.Crop(px=((0, 20), (0, 20), (0, 20), (0, 20)), keep_size=True)
                                         iaa.GaussianBlur(sigma=(.01, .2))
                                         ],
                                        random_order=False)

    def _load(self, image):
        img = imread(f"/media/ssd/train/{image}.jpg")
        # img = imread(f"../data/images/train/{image}.jpg")
        if self.shape != (180, 180):
            img = resize(img, self.shape)
        return img

    @threadsafe_generator
    def get_batch(self, batch_size):
        batch_size = int(batch_size)
        while True:
            idx = np.random.randint(0, self.num_images, batch_size)
            yield self._get_images(idx)

    def _get_images(self, idx_batch):
        X = np.array([self._load(self.images[idx]) for idx in idx_batch]).astype('float32')

        if self.aug:
            X = self.augmenter.augment_images(X)
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
    elif model_name == 'incres':
        base_model = InceptionResNetV2(include_top=False, input_shape=(150, 150, 3), pooling='avg')
        preprocess = preprocess_input
        shape = (150, 150)
    else:
        raise ValueError('Network name is undefined')

    x = base_model.output
    x = Dropout(.2)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Nadam(clipvalue=3, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, preprocess, shape


def hard_sampler(model, datagen, batch_size):
    logger.info('Sampler started')
    while True:
        samples, targets = [], []
        while len(samples) < batch_size:
            x_data, y_data = next(datagen)
            preds = model.predict(x_data)
            errors = np.abs(preds - y_data).max(axis=-1) > .99
            samples += x_data[errors].tolist()
            targets += y_data[errors].tolist()

        regular_samples = batch_size * 2 - len(samples)
        x_data, y_data = next(datagen)
        samples += x_data[:regular_samples].tolist()
        targets += y_data[:regular_samples].tolist()

        samples, targets = map(np.array, (samples, targets))

        idx = np.arange(batch_size * 2)
        np.random.shuffle(idx)
        batch1, batch2 = np.split(idx, 2)
        yield samples[batch1], targets[batch1]
        yield samples[batch2], targets[batch2]


def fit_model(model_name, batch_size=64, n_fold=0, use_hard_samples=False):

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

    if use_hard_samples:
        x, y = next(train.get_batch(batch_size))
        model.predict(x)
        # for some mysterious reasons it fails without this run if the model has just been loaded
        # ValueError: Tensor Tensor("dense_1_1/Softmax:0", shape=(?, 5270), dtype=float32) is not an element of this graph.

        logger.info('Switching to hard sampler')
        model.fit_generator(hard_sampler(model, train.get_batch(batch_size), batch_size=batch_size),
                            epochs=500,
                            steps_per_epoch=2000,
                            validation_data=val.get_batch(batch_size),
                            workers=1,
                            use_multiprocessing=False,
                            validation_steps=100,
                            initial_epoch=frozen_epochs + 50,
                            )


if __name__ == '__main__':
    Fire(fit_model)
