from keras.models import Model, Input, load_model, model_from_json
from layers import ReflectionPad
from layers import AdaIN, LRN, preprocess_symbolic_input
import numpy as np
from keras.engine import Layer
import keras.backend as K
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Lambda, Dropout
from keras.applications.imagenet_utils import preprocess_input

from skimage.io import imread, imsave
from tqdm import tqdm
from skimage.transform import resize
from keras.backend import tf as ktf
from keras.regularizers import l2

class Dummy(Layer):
    def __init__(self, shape, initializer='glorot_uniform', regularizer=None, activation='none', **kwargs):
        assert activation in ['none', 'exp']
        super(Dummy, self).__init__(**kwargs)
        self.shape = shape
        self.initializer = initializers.get(initializer)
        self.activation = activation
        self.regularizer = regularizer

    def build(self, input_shape):
        self.value = self.add_weight('var', self.shape, initializer=self.initializer,
                                     regularizer=self.regularizer)
        super(Dummy, self).build(input_shape=input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[0:1]) + list(self.shape))

    def call(self, inputs):
        if self.activation == 'none':
            val = self.value
        else:
            val = K.exp(self.value)

        return K.reshape(K.tile(val, K.shape(inputs)[0:1]), [-1,] + list(self.shape))

    def get_config(self):
        config = {'shape': self.shape, 'activation': self.activation}
        base_config = super(Dummy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomPerturbImage(Layer):
    def __init__(self, max_sat=0.05, **kwargs):
        super(RandomPerturbImage, self).__init__(**kwargs)
        self.max_sat = max_sat

    def call(self, inputs):
        img = ktf.cond(ktf.equal(K.learning_phase(), 0), lambda: inputs,
                        lambda: inputs + K.random_uniform(K.shape(inputs), -self.max_sat, self.max_sat))
        return img


def style_transfer_model(encoder='models/vgg_normalised.h5', decoder='models/decoder.h5', alpha = 0.5):
    from keras import backend as K

    K.set_image_data_format('channels_first')

    vgg = load_model(encoder, custom_objects={'ReflectionPad': ReflectionPad}, compile=False)

    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    y = outputs_dict['Threshold_30']

    encoder = Model(inputs=vgg.input, outputs=y, name='vgg16')
    encoder.trainable = False

    image = Input((3, 256, 256))

    style_mean = Dummy(shape=(512, ), initializer='zeros')
    style_var = Dummy(shape=(512, ), activation='exp', initializer='ones')

    m = Dropout(0.00)(style_mean(image))
    v = Dropout(0.00)(style_var(image))

    f_image = encoder(image)

    re_norm_image = AdaIN()([f_image, m, v])

    re_norm_image = Lambda(lambda inputs: alpha * inputs[0] + (1 - alpha) * inputs[1]) ([f_image, re_norm_image])

    decoder = load_model(decoder, custom_objects={'ReflectionPad':ReflectionPad})
    decoder.trainable = False
    out = decoder(re_norm_image)

    out = RandomPerturbImage()(out)
    out = Lambda(lambda x: K.clip(x, 0, 1))(out)
    st_model = Model(inputs=[image], outputs=[out])

    # style_mean.set_weights([np.mean(predicted, axis=(0, 2, 3))])
    # var = np.var(predicted, axis=(0, 2, 3))
    # var = np.log(var)
    # style_var.set_weights([var])

    return st_model

def blue_score(y_true, y_pred):
    return -K.mean(y_pred[:, 2] - (y_pred[:, 1] + y_pred[:, 0]) / 3)

def mem_model(memnet='memnet.h5'):
    memnet = load_model(memnet, custom_objects={'LRN': LRN}, compile=True)
    return memnet

def mem_score(y_true, y_pred):
    y_pred *= 255
    y_pred = ktf.transpose(y_pred, [0, 2, 3, 1])
    y_pred = ktf.image.resize_images(y_pred, (227, 227), )
    y_pred = ktf.transpose(y_pred, [0, 3, 1, 2])
    y_pred = preprocess_symbolic_input(y_pred, 'channels_first', 'caffe')
    memnet = mem_model()

    return -K.mean(memnet(y_pred))

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

    img = imread('dog.jpg')
    img = resize(img, (256, 256), preserve_range=True)
    img = np.array([img])
    train_set = preprocess_input(img)

    # img = train_set * 255
    # img = img - np.array([103.939, 116.779, 123.68]).reshape((1, -1, 1, 1))
    # img = img[:, ::-1, :, :]

    model = style_transfer_model(alpha=0.999)

    model.compile(Adam(lr=0.1), loss=mem_score)

    y = np.zeros((1, 1, 1, 1))
    for i in tqdm(range(1000)):
        if i % 100 == 0:
            img = deprocess_input(model.predict_on_batch(train_set))
            imsave('img%s.jpg' % i, img)
        score = model.train_on_batch(train_set, y)
        if i % 100 == 0:
            print score
    K.set_learning_phase(0)
    img = deprocess_input(model.predict_on_batch(train_set))
    imsave('img_res.jpg', img)

if __name__ == "__main__":
    main()
