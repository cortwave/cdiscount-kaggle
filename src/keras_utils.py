from keras import backend as K


def hard_bce(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred) * K.round(K.abs(y_true - y_pred) + .1),
                  axis=-1)
