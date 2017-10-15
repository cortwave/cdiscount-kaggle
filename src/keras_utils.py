from threading import Lock

from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import backend as K

import numpy as np


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


class AdamAccum(Optimizer):
    '''Adam accumulate optimizer.

    Default parameters follow those provided in the original paper. Wait for several mini-batch to update

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, accum_iters=10, **kwargs):
        super(AdamAccum, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.accum_iters = K.variable(accum_iters)
        self.epsilon = epsilon

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        vs = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        gs = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = ms + vs

        for p, g, m, v, gg in zip(params, grads, ms, vs, gs):

            flag = K.equal(self.iterations % self.accum_iters, 0)
            flag = K.cast(flag, dtype='float32')

            gg_t = (1 - flag) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / self.accum_iters
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((gg + flag * g) / self.accum_iters)
            p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, flag * m_t + (1 - flag) * m))
            self.updates.append((v, flag * v_t + (1 - flag) * v))
            self.updates.append((gg, gg_t))

            new_p = p_t
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(AdamAccum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))