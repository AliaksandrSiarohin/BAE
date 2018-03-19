from keras.backend import tf as ktf
from keras.engine import Layer
from keras.utils import conv_utils
import keras.backend as K

class ReflectionPad(Layer):
    def __init__(self, padding=((1, 1), (1, 1)), data_format=None, **kwargs):
        super(ReflectionPad, self).__init__(**kwargs)
        self.paddings = [list(p) for p in padding]
        self.data_format = conv_utils.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        if input_shape[2] is None:
            return (input_shape[0], None, None, input_shape[3]) if self.data_format == 'channels_last' else (
                input_shape[0], input_shape[1], None, None)
        if self.data_format == 'channels_last':
            shape = (input_shape[0], input_shape[1] + sum(self.paddings[0]), input_shape[2] + sum(self.paddings[1]),
                     input_shape[3])
        else:
            shape = (input_shape[0], input_shape[1], input_shape[2] + sum(self.paddings[0]),
                     input_shape[3] + sum(self.paddings[1]))

        return shape

    def call(self, inputs):
        if self.data_format == 'channels_last':
            out = ktf.pad(inputs, [[0, 0]] + self.paddings + [[0, 0]], mode='REFLECT')
        else:
            out = ktf.pad(inputs, [[0, 0], [0, 0]] + self.paddings, mode='REFLECT')

        return out

    def get_config(self):
        config = {'padding': self.paddings}
        base_config = super(ReflectionPad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaIN(Layer):
    def __init__(self, data_format=None, eps=1e-7, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.spatial_axis = [1, 2] if self.data_format == 'channels_last' else [2, 3]
        self.eps = eps

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs):
        image = inputs[0]
        if len(inputs) == 2:
            style = inputs[1]
            style_mean, style_var = ktf.nn.moments(style, self.spatial_axis, keep_dims=True)
        else:
            style_mean = ktf.expand_dims(ktf.expand_dims(inputs[1], self.spatial_axis[0]), self.spatial_axis[1])
            style_var = ktf.expand_dims(ktf.expand_dims(inputs[2], self.spatial_axis[0]), self.spatial_axis[1])
        image_mean, image_var = ktf.nn.moments(image, self.spatial_axis, keep_dims=True)
        out = ktf.nn.batch_normalization(image, image_mean,
                                         image_var, style_mean,
                                         ktf.sqrt(style_var), self.eps)
        return out

    def get_config(self):
        config = {'eps': self.eps}
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


import numpy as np
def preprocess_symbolic_input(x, data_format, mode, IMAGENET_MEAN=None):
    IMAGENET_MEAN = None

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if IMAGENET_MEAN is None:
        IMAGENET_MEAN = K.constant(-np.array([103.939, 116.779, 123.68]))
    # Zero-center by mean pixel
    if K.dtype(x) != K.dtype(IMAGENET_MEAN):
        x = K.bias_add(x, K.cast(IMAGENET_MEAN, K.dtype(x)), data_format)
    else:
        x = K.bias_add(x, IMAGENET_MEAN, data_format)

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if K.ndim(x) == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]


    return x

class LRN(Layer):
    def __init__(self, alpha=1e-4, k=1, beta=0.75, n=5, data_format=None, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        super(LRN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, X, mask=None):
        if self.data_format == 'channels_last':
            b, r, c, ch = K.int_shape(X)
        else:
            b, ch, r, c = K.int_shape(X)
        half = self.n // 2
        square = K.square(X)
        if self.data_format != 'channels_last':
            extra_channels = K.spatial_2d_padding(square,
                                                  padding=((half, half), (0, 0)), data_format='channels_last')
        else:
            extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 3, 1, 2)),
                                                  padding=((half, half), (0, 0)), data_format='channels_last')

        scale = self.k
        for i in range(self.n):
            scale += (self.alpha / self.n) * extra_channels[:, i:(i + ch), :, :]
        scale = scale ** self.beta
        if self.data_format == 'channels_last':
            scale = K.permute_dimensions(scale, (0, 2, 3, 1))
        return X / scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {"name": self.name,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n,
                "data_format": self.data_format}
