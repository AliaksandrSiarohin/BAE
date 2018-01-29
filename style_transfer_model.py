from keras.models import Model, Input, load_model
from layers import ReflectionPad
from layers import AdaIN
import numpy as np
from keras import backend as K


def get_encoder(encoder='models/vgg_normalised.h5'):
    K.set_image_data_format('channels_first')

    vgg = load_model(encoder, custom_objects={'ReflectionPad': ReflectionPad}, compile=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    y = outputs_dict['Threshold_30']

    encoder = Model(inputs=vgg.input, outputs=y, name='vgg16')
    inp = Input((3, None, None))
    return Model(inputs=inp, outputs=encoder(inp))


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

    decoder.summary()

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


