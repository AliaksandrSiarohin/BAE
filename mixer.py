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
        self.style_tr_model = style_transfer_gan(encoder=self.args.encoder, decoder=self.args.decoder,
                                                   style_generator=self.args.style_generator,
                                                   alpha=self.args.alpha, z_style_shape=(self.args.z_style_dim, ),
                                                   image_shape=self.args.image_shape)
        z_style = K.placeholder(shape=(self.args.z_style_dim, ))
        img_tensor = K.placeholder(shape=(1,) + self.args.image_shape)
        stylized_image = self.style_tr_model([img_tensor, K.reshape(z_style, (1, self.args.z_style_dim))])

        score = get_score(stylized_image,
                          z_style, image_score_fun=self.args.score_type, weight_image=self.args.weight_image)
        grad = K.gradients(K.sum(score), z_style)

        self.f = K.function(inputs=[z_style, K.learning_phase(), img_tensor], outputs=[K.sum(score), grad[0]])


    def run(self, image, verbose=False, seed = None):
        img = resize(image, self.args.image_shape[1:], preserve_range=True)
        img = img[np.newaxis]
        content_image = preprocess_input(img)

        oracle = lambda z: self.f([z, 0, content_image])
        if seed is not None:
            np.random.seed(seed)
        self.gd = self.args.optimizer(oracle, self.args.lr)
        self.gd.initialize(np.random.normal(size=(64, )))

        generated_images = []

        iters = tqdm(range(self.args.number_of_iters)) if verbose else range(self.args.number_of_iters)

        for _ in iters:
            raw_image = self.style_tr_model.predict_on_batch([content_image,
                                                              np.expand_dims(self.gd.current, axis=0)])
            processed_image = deprocess_input(raw_image)
            generated_images.append(processed_image)
            self.gd.update()

        return generated_images
