from keras.models import Model, Input, load_model
import keras.backend as K
from keras.layers import Lambda

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
from skimage.transform import resize
from scipy.special import logit

import os
from argparse import ArgumentParser

from layers import ReflectionPad
from layers import AdaIN

from optimizers import GradientAccent, MetropolisHastingsMCMC, LangevinMCMC, HamiltonyanMCMC
from losses import get_score


def style_transfer_model(encoder='models/vgg_normalised.h5', decoder='models/decoder.h5',
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


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--encoder", default='models/vgg_normalised.h5', help='Path to the decoder for adain model')
    parser.add_argument("--decoder", default='models/decoder.h5', help="Path to the encoder for adain model")
    parser.add_argument("--style_generator", default='output/watercolor_ch/epoch_9999_generator.h5',
                        help="Path to generator trained using style_gan_train.py")

    parser.add_argument("--alpha", type=float, default=0.0, help="How much style of content image to preserve")
    parser.add_argument("--z_style_dim", type=int, default=64, help="Dimensionality of latent space of the gan")
    parser.add_argument("--image_shape", type=lambda x: map(int, x.split(',')), default=(3, 256, 256),
                        help="Shape of the resulting image")
    parser.add_argument("--lr", type=float, default=0.1,
         help="Learning rate for Gradient Accent, tao for Langevein Dynamycs, transition var for MetropolisHastings")
    parser.add_argument("--optimizer", type=str, default='hamiltonyan', choices=['grad', 'langevin', 'hamiltonyan', 'mh'])
    parser.add_argument("--content_image", default='cornell_cropped.jpg')

    parser.add_argument("--samples_dir", default='output/stylized_images')
    parser.add_argument("--display_ratio", type=int, default=10)
    parser.add_argument("--number_of_epochs", type=int, default=100)
    parser.add_argument("--score_type", choices=['blue', 'mem'], default='mem',
                        help="Score type 'blue' is making image more blue, 'mem' - make an image more memoreble")
    parser.add_argument("--weight_image", type=float, default=50, help='Weight of the image score')

    optimizers = {'grad': GradientAccent, 'langevin': LangevinMCMC, 'mh': MetropolisHastingsMCMC, 'hamiltonyan':HamiltonyanMCMC}
    args = parser.parse_args()
    args.optimizer = optimizers[args.optimizer]
    return args


def print_image_score(score, score_type):
    if score_type == 'blue':
        print ("Image blue score %s" % score)
    elif score_type == 'mem':
        score = logit(np.exp(score))
        print ("Image memorability score %s" % score)
    return score


def main():
    args = parse_args()
    z_style_shape = (args.z_style_dim, )

    img = imread(args.content_image)
    img = resize(img, args.image_shape[1:], preserve_range=True)
    img = np.array([img])
    content_image = preprocess_input(img)

    model = style_transfer_model(encoder=args.encoder, decoder=args.decoder,
                                 style_generator=args.style_generator,
                                 alpha=args.alpha, z_style_shape=z_style_shape, image_shape=args.image_shape)

    z_style = K.placeholder(shape=z_style_shape)
    img_tensor = K.constant(content_image / 255.0)
    stylized_image = model([img_tensor, K.reshape(z_style, (1, ) + z_style_shape)])
    score = get_score(stylized_image,
                      z_style, image_score_fun=args.score_type, weight_image=args.weight_image)
    grad = K.gradients(K.sum(score), z_style)

    f = K.function(inputs=[z_style, K.learning_phase()], outputs=[K.sum(score), grad[0]])

    f_metrics = K.function(inputs=[z_style, K.learning_phase()], outputs=[score])
    metrics_fun = lambda z: f_metrics([z, 0])

    score_image = K.function(inputs=[img_tensor, z_style, K.learning_phase()],
        outputs=[get_score(img_tensor, z_style, image_score_fun=args.score_type, weight_image=args.weight_image)[1]])

    oracle = lambda z: f([z, 0])

    gd = args.optimizer(oracle, args.lr)
    gd.initialize(np.random.normal(size=(64, )))

    if not os.path.exists(args.samples_dir):
        os.makedirs(args.samples_dir)

    initial_image_score = score_image([content_image, gd.current, 0])[0]
    initial_image_score /= args.weight_image
    print_image_score(initial_image_score, args.score_type)

    for i in tqdm(range(args.number_of_epochs + 1)):
        gd.update()
        if i % args.display_ratio == 0:
            unprocesed_img = model.predict_on_batch([content_image, np.expand_dims(gd.current, axis=0)])
            img = deprocess_input(unprocesed_img)
            imsave(os.path.join(args.samples_dir, 'img%s.jpg') % i, img)
            prior, image_score = metrics_fun(gd.current)[0]
            print_image_score(image_score / args.weight_image, args.score_type)
            print ("Sum %s, prior %s, value %s" % (prior + image_score, prior, image_score))

if __name__ == "__main__":
    main()
