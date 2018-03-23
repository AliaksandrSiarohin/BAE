from style_transfer_model import style_transfer_gan, preprocess_input, deprocess_input
from losses import get_score
import keras.backend as K

import numpy as np
from skimage.transform import resize

from tqdm import tqdm

class ChainMix(object):
    def __init__(self, args):
        self.args = args

    def compile(self):
        if self.args.alpha_sigma != 0:
            z_style_and_alpha = K.placeholder(shape=(self.args.z_style_dim + 1, ))
            alpha = z_style_and_alpha[-1]
            z_style = z_style_and_alpha[:-1]
        else:
            z_style_and_alpha = K.placeholder(shape=(self.args.z_style_dim, ))
            z_style = z_style_and_alpha
            alpha = self.args.alpha_mean

        img_tensor = K.placeholder(shape=(1,) + self.args.image_shape)

        stylized_image = style_transfer_gan(image=img_tensor, z_style=K.expand_dims(z_style, axis=0), alpha=alpha,
                                            encoder=self.args.encoder, decoder=self.args.decoder,
                                            style_generator=self.args.style_generator)

        score = get_score(stylized_image, z_style, alpha,
                          alpha_mean=self.args.alpha_mean, alpha_sigma=self.args.alpha_sigma,
                          image_score_fun=self.args.score_type, weight_image=self.args.weight_image)

        grad = K.gradients(K.sum(score), z_style_and_alpha)[0]

        updates = []
        if self.args.adaptive_grad:
            beta = 0.99
            v = K.zeros(K.int_shape(grad))
            t = K.ones(shape=())
            v_t = (beta * v) + (1. - beta) * K.square(grad)
            vupdate = K.update(v, v_t)
            tupdate = K.update_add(t, 1)
            v_hat = v_t / (1 - beta ** t)
            updates.append(vupdate)
            updates.append(tupdate)
            grad /= K.sqrt(v_hat)

        self.f = K.function(inputs=[z_style_and_alpha, K.learning_phase(), img_tensor],
                            outputs=[K.sum(score), grad], updates=updates)

        self.st = K.function(inputs=[z_style_and_alpha, K.learning_phase(), img_tensor],
                             outputs=[stylized_image])


    def run(self, image, verbose=False, seed = None):
        img = resize(image, self.args.image_shape[1:], preserve_range=True)
        content_image = preprocess_input(img)

        oracle = lambda z: self.f([z, 0, content_image])
        if seed is not None:
            np.random.seed(seed)
        self.gd = self.args.optimizer(oracle, self.args.lr, self.args.lr_decay)
        z_init = np.random.normal(size=self.args.z_style_dim)
        alpha_init = np.array([self.args.alpha_mean])

        if self.args.alpha_sigma != 0:
            self.gd.initialize(np.concatenate([z_init, alpha_init], axis=0))
        else:
            self.gd.initialize(z_init)

        generated_images = []
        alphas = []

        iters = tqdm(range(self.args.number_of_iters)) if verbose else range(self.args.number_of_iters)

        for _ in iters:
            raw_image = self.st([self.gd.current, 0, content_image])[0]
            alpha = self.args.alpha_mean if self.args.alpha_sigma == 0 else self.gd.current[-1]
            alphas.append(alpha)
            processed_image = deprocess_input(raw_image)
            generated_images.append(processed_image)
            self.gd.update()

        return generated_images, alphas
