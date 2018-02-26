from keras.models import Model, Input, load_model
from layers import ReflectionPad
from layers import AdaIN
from keras.layers import Lambda

import numpy as np
from keras import backend as K
from skimage.color import gray2rgb


def style_transfer_gan(encoder='models/vgg_normalised.h5', decoder='models/decoder.h5',
                         style_generator='output/pen_ink_ch/epoch_1999_generator.h5',
                         alpha=0.5, z_style_shape=(64, ), image_shape=(3, 256, 256)):
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
    v = Lambda(lambda st: K.exp(st[:, 512:]), output_shape=(512, ))(style)

    f_image = encoder(image)

    re_norm_image = AdaIN()([f_image, m, v])

    re_norm_image = Lambda(lambda inputs: alpha * inputs[0] + (1 - alpha) * inputs[1])([f_image, re_norm_image])

    decoder = load_model(decoder, custom_objects={'ReflectionPad':ReflectionPad})
    decoder.trainable = False
    out = decoder(re_norm_image)

    out = Lambda(lambda x: K.clip(x, 0, 1))(out)
    st_model = Model(inputs=[image, z_style], outputs=[out], name='style_transfer_model')

    return st_model


def get_encoder(encoder='models/vgg_normalised.h5'):
    K.set_image_data_format('channels_first')

    vgg = load_model(encoder, custom_objects={'ReflectionPad': ReflectionPad}, compile=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    y = outputs_dict['Threshold_30']

    encoder = Model(inputs=vgg.input, outputs=y, name='vgg16')
    inp = Input((3, None, None))
    return Model(inputs=inp, outputs=encoder(inp), name='vgg_arb_inp')


def get_decoder(decoder='models/decoder.h5'):
    K.set_image_data_format('channels_first')
    decoder = load_model(decoder, custom_objects = {'ReflectionPad' : ReflectionPad})
    return decoder


def style_transfer_model(encoder='models/vgg_normalised.h5', decoder='models/decoder.h5'):
    encoder = get_encoder(encoder)

    image, style = Input((3, None, None)), Input((3, None, None))

    f_image = encoder(image)
    f_style = encoder(style)

    re_norm_image = AdaIN()([f_image, f_style])

    decoder = get_decoder(decoder)
    out = decoder(re_norm_image)

    st_model = Model(inputs=[image, style], outputs=out)

    return st_model


def preprocess_input(inp):
    if len(inp.shape) == 2:
        inp = gray2rgb(inp)
    if len(inp.shape) == 3:
        inp = inp[np.newaxis]
    out = np.moveaxis(inp, -1, 1).astype('float') / 255.0
    return out


def deprocess_input(inp):
    out = np.clip(255 * inp, 0, 255)
    out = np.moveaxis(out, 1, -1)
    out = np.squeeze(out)
    return out.astype(np.uint8)


