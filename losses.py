import keras.backend as K
from keras.models import load_model
from keras.backend import tf as ktf

from layers import preprocess_symbolic_input, LRN
import numpy as np

def alpha_prior(alpha, alpha_mean, alpha_sigma):
    if alpha_sigma == 0:
        return 0
    elif alpha_sigma == float('inf'):
        alpha_between01 = ktf.logical_and(ktf.greater(alpha, 0.0), ktf.less(alpha, 1.0))
        return ktf.cond(alpha_between01, lambda: 0.0, lambda: -float('inf')) 
    else:
        return -0.5 * K.sum((alpha - alpha_mean) * (alpha - alpha_mean)) / (alpha_sigma ** 2)

def gaussian_prior(z_style):
    return -0.5 * K.sum(z_style * z_style)


def blue_score(stylized_image):
    return K.log(K.mean(stylized_image[:, 2] - (stylized_image[:, 1] + stylized_image[:, 0]) / 3))


def get_image_memorability(image, net='models/mem_internal.h5'):
    memnet = load_model(net, custom_objects={'LRN': LRN}, compile=True)

    image *= 255
    image = ktf.transpose(image, [0, 2, 3, 1])
    image = ktf.image.resize_images(image, (227, 227), )
    image = ktf.transpose(image, [0, 3, 1, 2])
    image = preprocess_symbolic_input(image, data_format='channels_first', mode='caffe')

    score = memnet(image)
    return score


def mem_score(stylized_image, memnet='models/mem_internal.h5'):
    stylized_score = get_image_memorability(stylized_image, net=memnet)
    return K.log(K.sigmoid(K.sum(stylized_score)))


def aes_score(stylized_image, ilgnet='models/ava1.h5', ava1_mean=True):
    stylized_score = get_image_aesthetics(stylized_image, net=ilgnet, ava1_mean=ava1_mean)
    return K.log(K.sum(stylized_score))


def get_image_aesthetics(image, net='models/ava1.h5', ava1_mean=True):
    ilgnet = load_model(net, custom_objects={'LRN': LRN}, compile=True)

    if not ava1_mean:
        IMAGENET_MEAN = K.constant(-np.array([90.05038554687053, 98.58527740451973, 107.50910979753826])[::-1])
    else:
        IMAGENET_MEAN = K.constant(-np.array([91.00178961467464, 99.46385127608664, 107.8456162486691])[::-1])

    image *= 255
    image = ktf.transpose(image, [0, 2, 3, 1])
    image = ktf.image.resize_images(image, (227, 227), )
    image = ktf.transpose(image, [0, 3, 1, 2])
    image = preprocess_symbolic_input(image, data_format='channels_first', mode='caffe',
                                      IMAGENET_MEAN=IMAGENET_MEAN)

    score = ilgnet(image)
    return score[:, 1]


def get_score(stylized_image, z_style, alpha, alpha_mean=0.5, alpha_sigma=0,
              weight_image=20, weight_prior=1, image_score_fun='blue', **kwargs):
    assert image_score_fun in ['blue', 'mem', 'aes']
    score_funs = {'blue': blue_score, 'mem': mem_score, 'aes': aes_score}
    gp = weight_prior * gaussian_prior(z_style)
    isf = weight_image * score_funs[image_score_fun](stylized_image, **kwargs)
    ap = alpha_prior(alpha, alpha_mean, alpha_sigma)
    return ktf.stack([gp, isf, ap])
