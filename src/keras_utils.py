from threading import Lock

from keras import backend as K


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


def hard_bce(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred) * K.round(K.abs(y_true - y_pred) + .1),
                  axis=-1)
