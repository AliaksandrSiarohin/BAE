import keras.backend as K
from keras.models import load_model
from keras.backend import tf as ktf

from layers import preprocess_symbolic_input, LRN


def gaussian_prior(z_style, weight=1):
    return 0.5 * weight * K.sum(z_style * z_style)


def blue_score(stylized_image, weight=20):
    return -weight * K.log(K.mean(stylized_image[:, 2] - (stylized_image[:, 1] + stylized_image[:, 0]) / 3))


def mem_model(memnet='memnet.h5'):
    memnet = load_model(memnet, custom_objects={'LRN': LRN}, compile=True)
    return memnet


def mem_score(stylized_image, weight=1):
    stylized_image *= 255
    stylized_image = ktf.transpose(stylized_image, [0, 2, 3, 1])
    stylized_image = ktf.image.resize_images(stylized_image, (227, 227), )
    stylized_image = ktf.transpose(stylized_image, [0, 3, 1, 2])
    stylized_image = preprocess_symbolic_input(stylized_image, 'channels_first', 'caffe')
    memnet = mem_model()
    return -K.mean(K.log(K.sigmoid(memnet(stylized_image))))


def get_score(stylized_image, z_style, weight_image=20, weight_prior=1, image_score_fun='blue'):
    assert image_score_fun in ['blue', 'mem']
    score_funs = {'blue': blue_score, 'mem': mem_score}
    gp = gaussian_prior(z_style, weight_prior)
    isf = score_funs[image_score_fun](stylized_image, weight_image)
    return gp + isf
