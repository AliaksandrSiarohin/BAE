from keras.backend import tf as ktf
from keras.engine import Layer
from keras.utils import conv_utils


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
