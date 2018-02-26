from argparse import ArgumentParser
from optimizers import GradientAccent, LangevinMCMC, MetropolisHastingsMCMC, HamiltonyanMCMC


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--encoder", default='models/vgg_normalised.h5', help='Path to the decoder for adain model')
    parser.add_argument("--decoder", default='models/decoder.h5', help="Path to the encoder for adain model")
    parser.add_argument("--style_generator", default='output/devian_art/epoch_9999_generator.h5',
                        help="Path to generator trained using style_gan_train.py")

    parser.add_argument("--alpha", type=float, default=0.0, help="How much style of content image to preserve")
    parser.add_argument("--z_style_dim", type=int, default=64, help="Dimensionality of latent space of the gan")
    parser.add_argument("--image_shape", type=lambda x: map(int, x.split(',')), default=(3, 256, 256),
                        help="Shape of the resulting image")
    parser.add_argument("--lr", type=float, default=0.1,
         help="Learning rate for Gradient Accent, tao for Langevein Dynamycs, transition var for MetropolisHastings")
    parser.add_argument("--optimizer", type=str, default='langevin', choices=['grad', 'langevin', 'hamiltonyan', 'mh'])
    parser.add_argument("--content_image", default='cornell_cropped.jpg')

    parser.add_argument("--output_dir", default='output/memorability-evaluation')
    parser.add_argument("--display_ratio", type=int, default=1)
    parser.add_argument("--number_of_iters", type=int, default=500)
    parser.add_argument("--score_type", choices=['blue', 'mem'], default='mem',
                        help="Score type 'blue' is making image more blue, 'mem' - make an image more memoreble")
    parser.add_argument("--weight_image", type=float, default=100, help='Weight of the image score')

    optimizers = {'grad': GradientAccent, 'langevin': LangevinMCMC,
                  'mh': MetropolisHastingsMCMC, 'hamiltonyan': HamiltonyanMCMC}

    parser.add_argument("--samples_dir", default='output/generated_devian_art_langevin')

    parser.add_argument("--content_images_folder", default='../memorability/lamem/images/', help="Content images for evaluation")
    parser.add_argument("--content_images_names_file", default='dataset/memorability_test_images.txt', help="File with content image names")
    # parser.add_argument("--content_images_folder", default='dataset/tmp', help="Content images for evaluation")
    # parser.add_argument("--content_images_names_file", default='dataset/tmp.txt', help="File with content image names")
    parser.add_argument("--external_scorer", default='models/mem_external.h5', help='External memorability scorer')
    parser.add_argument("--internal_scorer", default='models/mem_internal.h5', help="Internal memorability scorer")
    parser.add_argument("--styles_images_dir", default='dataset/devian_art',
                                        help='Directory with styles for baseline evaluation')

    args = parser.parse_args()

    args.optimizer = optimizers[args.optimizer]

    return args
