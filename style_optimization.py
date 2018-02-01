from keras.models import Model, Input, load_model
from keras.engine import Layer
import keras.backend as K
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Lambda
from keras.backend import tf as ktf
from keras.regularizers import Regularizer

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
from skimage.transform import resize

from layers import ReflectionPad
from layers import AdaIN, LRN, preprocess_symbolic_input

from optimizers import GradientDecent
from losses import get_score

# class GausianPrior(Regularizer):
#     """Regularizer base class.
#     """
#     def __call__(self, x):
#         dim = K.int_shape(x)[0]
#         log = 0.5 * K.sum(x * x)
# #        log -= (dim / 2) * np.log(2 * np.pi)
#         return log
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#
#
# class Dummy(Layer):
#     def __init__(self, shape, initializer='glorot_uniform', regularizer=None, activation='none', **kwargs):
#         assert activation in ['none', 'exp']
#         super(Dummy, self).__init__(**kwargs)
#         self.shape = shape
#         self.initializer = initializers.get(initializer)
#         self.activation = activation
#         self.regularizer = regularizer
#
#     def build(self, input_shape):
#         self.value = self.add_weight('var', self.shape, initializer=self.initializer,
#                                      regularizer=self.regularizer)
#         super(Dummy, self).build(input_shape=input_shape)
#
#     def compute_output_shape(self, input_shape):
#         return tuple(list(input_shape[0:1]) + list(self.shape))
#
#     def call(self, inputs):
#         if self.activation == 'none':
#             val = self.value
#         else:
#             val = K.exp(self.value)
#
#         return K.reshape(K.tile(val, K.shape(inputs)[0:1]), [-1,] + list(self.shape))
#
#     def get_config(self):
#         config = {'shape': self.shape, 'activation': self.activation}
#         base_config = super(Dummy, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def style_transfer_model(encoder='models/vgg_normalised.h5', decoder='models/decoder.h5',
                         style_generator='output/checkpoints/epoch_9999_generator.h5',
                         alpha=0.5, z_style_shape=(64, ), image_shape = (3, 256, 256)):
    from keras import backend as K

    K.set_image_data_format('channels_first')

    vgg = load_model(encoder, custom_objects={'ReflectionPad': ReflectionPad}, compile=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    y = outputs_dict['Threshold_30']

    encoder = Model(inputs=vgg.input, outputs=y, name='vgg16')
    encoder.trainable = False

    image = Input(image_shape)
    z_style = Input(z_style_shape)

    style_generator = load_model(style_generator, compile=False)
    style_generator.name = 'style_generator'
    style_generator.trainable = False
    style = style_generator(z_style)

    m = Lambda(lambda st: st[:, :512], output_shape=(512, ))(style)
    v = Lambda(lambda st: st[:, 512:], output_shape=(512, ))(style)

    f_image = encoder(image)

    re_norm_image = AdaIN()([f_image, m, v])

    re_norm_image = Lambda(lambda inputs: alpha * inputs[0] + (1 - alpha) * inputs[1])([f_image, re_norm_image])

    decoder = load_model(decoder, custom_objects={'ReflectionPad':ReflectionPad})
    decoder.trainable = False
    out = decoder(re_norm_image)

    out = Lambda(lambda x: K.clip(x, 0, 1))(out)
    st_model = Model(inputs=[image, z_style], outputs=[out], name='style_transfer_model')

    return st_model


def preprocess_input(inp):
    if len(inp.shape) == 3:
        inp = inp[np.newaxis]
    out = np.moveaxis(inp, -1, 1).astype('float') / 255.0
    return out


def deprocess_input(inp):
    out = np.clip(255 * inp, 0, 255)
    out = np.moveaxis(out, 1, -1)
    out = np.squeeze(out)
    return out.astype(np.uint8)



def main():
    K.set_learning_phase(1)

    img = imread('cornell_cropped.jpg')
    img = resize(img, (256, 256), preserve_range=True)
    img = np.array([img])
    train_set = preprocess_input(img)

    model = style_transfer_model(alpha=0)
    _, z_style = model.input
    img_tensor = K.constant(train_set)
    stylized_image = model([img_tensor, z_style])
    score = get_score(stylized_image, z_style)
    grad = K.gradients(score, z_style)

    f = K.function([z_style, K.learning_phase()], [score, grad])
    oracle = lambda z: f([z, 0])

    gd = GradientDecent(oracle)
    gd.initialize(np.random.normal(size=(1, 64)))

    for i in tqdm(range(1000)):
        if i % 100 == 0:
            img = deprocess_input(model.predict_on_batch([train_set, gd.current]))
            imsave('img%s.jpg' % i, img)
        score = gd.update()
        if i % 100 == 0:
            print score
    K.set_learning_phase(0)
    img = deprocess_input(model.predict_on_batch([train_set, gd.current]))
    imsave('img_res.jpg', img)

if __name__ == "__main__":
    main()
