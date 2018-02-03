from keras.models import Input, Sequential, Model
from keras.layers import Dense, Concatenate, BatchNormalization

from keras.layers.advanced_activations import LeakyReLU


import numpy as np
import os
from skimage.io import imread
from skimage.color import gray2rgb

from gan.wgan_gp import WGAN_GP
from gan.dataset import ArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer

from style_transfer_model import get_encoder, get_decoder, preprocess_input, deprocess_input
from layers import AdaIN

from tqdm import tqdm


def make_generator(number_of_channels=512, noise_size=64):
    inp = Input((noise_size,))

    out = Dense(128, activation='relu')(inp)
    out = Dense(512, activation='relu')(out)

    mean = Dense(512, activation=None)(out)
    var = Dense(512, activation=None)(out)

    out = Concatenate(axis=-1)([mean, var])

    return Model(inputs=inp, outputs=out)


def make_discriminator(number_of_channels=512):
    inp = Input((number_of_channels * 2,))

    out = Dense(512, activation='relu')(inp)
    out = Dense(256, activation='relu')(out)
    out = Dense(128, activation='relu')(out)
    out = Dense(1, activation=None)(out)

    return Model(inputs=inp, outputs=out)


class StylesDataset(ArrayDataset):
    def __init__(self, batch_size, input_dir, noise_size=(64,), cache_file_name=None,
                 content_image='cornell_cropped.jpg', number_of_channels=512):
        self.input_dir = input_dir
        self.cache_file_name = cache_file_name
        self.content_image = preprocess_input(imread(content_image))
        self.number_of_channels = number_of_channels

        X = self.extract_batch_statistics_from_images()
        self.st_model = self.create_style_transfer_model()
        super(StylesDataset, self).__init__(X, batch_size, noise_size)
        self._batches_before_shuffle = X.shape[0] // batch_size

        self._X[:, self.number_of_channels:] = np.log(self._X[:, self.number_of_channels:] + 1e-5)

    def extract_batch_statistics_from_images(self):
        style_encoder_model = get_encoder()
        print ("Extract batch statistics from content image...")
        self.content_image_emb = style_encoder_model.predict(self.content_image)

        ## Load from cache if it exist
        if self.cache_file_name is not None and os.path.exists(self.cache_file_name):
            X = np.load(self.cache_file_name)
            return X

        images_names = os.listdir(self.input_dir)
        X = []  # np.empty(shape=(len(images_names), self.number_of_channels * 2))

        print ("Extract batch statistics from style images...")
        for i, name in tqdm(list(enumerate(images_names))):
            try:
                img = imread(os.path.join(self.input_dir, name))
            except:
                print ("Error when processing image %s" % name)
                continue
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = gray2rgb(img)
            if img.shape[2] == 4:
                img = img[..., :3]
            img = preprocess_input(img)
            emb = style_encoder_model.predict(img)

            mean = np.mean(emb, axis=(0, 2, 3))
            var = np.var(emb, axis=(0, 2, 3))

            X.append(np.concatenate([mean, var], axis=0))

        X = np.array(X)
        if self.cache_file_name is not None:
            print ('Saving to batch statistics to cache...')
            cache_dir = os.path.dirname(self.cache_file_name)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            np.save(self.cache_file_name, X)
        print ("Shape of X is %s" % str(X.shape))
        return X

    def create_style_transfer_model(self):
        image = Input((self.number_of_channels, None, None))
        mean = Input((self.number_of_channels,))
        var = Input((self.number_of_channels,))

        re_norm_image = AdaIN()([image, mean, var])

        decoder = get_decoder()
        out = decoder(re_norm_image)
        st_model = Model(inputs=[image, mean, var], outputs=[out])

        return st_model

    def display(self, output_batch, input_batch=None, number_of_display_images=5):
        batch = output_batch

        def display_batch_of_images(batch):
            mean = batch[:, :self.number_of_channels]
            var = np.exp(batch[:, self.number_of_channels:])
            images = []

            for i in range(number_of_display_images):
                img = self.st_model.predict([self.content_image_emb, mean[i:(i + 1)], var[i:(i + 1)]])
                images.append(np.expand_dims(deprocess_input(img), axis=0))

            batch = np.concatenate(images, axis=0)
            image = super(StylesDataset, self).display(batch)
            return image

        generated = display_batch_of_images(batch)
        true = display_batch_of_images(self.next_discriminator_sample()[0])

        return np.concatenate([generated, true], axis=1)


def main():
    generator = make_generator()
    discriminator = make_discriminator()

    parser = parser_with_default_args()
    parser.add_argument("--input_dir", default='dataset/media_watercolor')
    parser.add_argument("--cache_file_name", default=None)
    parser.add_argument("--content_image", default='cornell_cropped.jpg')

    args = parser.parse_args()

    dataset = StylesDataset(args.batch_size, args.input_dir, cache_file_name=args.cache_file_name,
                            content_image=args.content_image)
    gan = WGAN_GP(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))

    trainer.train()


if __name__ == "__main__":
    main()
