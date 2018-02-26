import keras.backend as K
from keras.models import load_model
from keras.backend import tf as ktf

from layers import preprocess_symbolic_input, LRN


def gaussian_prior(z_style):
    return -0.5 * K.sum(z_style * z_style)


def blue_score(stylized_image):
    return K.log(K.mean(stylized_image[:, 2] - (stylized_image[:, 1] + stylized_image[:, 0]) / 3))


def get_image_memorability(image, memnet='models/mem_internal.h5'):
    get_image_memorability.mem_net = load_model(memnet, custom_objects={'LRN': LRN}, compile=True)

    memnet = get_image_memorability.mem_net

    image *= 255
    image = ktf.transpose(image, [0, 2, 3, 1])
    image = ktf.image.resize_images(image, (227, 227), )
    image = ktf.transpose(image, [0, 3, 1, 2])
    image = preprocess_symbolic_input(image, data_format='channels_first', mode='caffe')

    score = memnet(image)
    return score


def mem_score(stylized_image, memnet='models/mem_internal.h5'):
    stylized_score = get_image_memorability(stylized_image, memnet=memnet)
    return K.log(K.sigmoid(K.sum(stylized_score)))


def get_score(stylized_image, z_style, weight_image=20, weight_prior=1, image_score_fun='blue', **kwargs):
    assert image_score_fun in ['blue', 'mem']
    score_funs = {'blue': blue_score, 'mem': mem_score}
    gp = weight_prior * gaussian_prior(z_style)
    isf = weight_image * score_funs[image_score_fun](stylized_image, **kwargs)
    return ktf.stack([gp, isf])
