from argparse import ArgumentParser
from optimizers import GradientAccent, LangevinMCMC, MetropolisHastingsMCMC, HamiltonyanMCMC
from functools import partial

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--encoder", default='models/vgg_normalised.h5', help='Path to the decoder for adain model')
    parser.add_argument("--decoder", default='models/decoder.h5', help="Path to the encoder for adain model")
    parser.add_argument("--style_generator", default='output/devian_art/epoch_9999_generator.h5',
                        help="Path to generator trained using style_gan_train.py")

    parser.add_argument("--alpha_sigma", type=float, default=float('inf'), help="How much style of content image to preserve."
                                                                       "Std value, set to zero if not used.")
    parser.add_argument("--alpha_mean", type=float, default=0.5, help="How much style of content image to preserve."
                                                                      "Mean value.")

    parser.add_argument("--z_style_dim", type=int, default=64, help="Dimensionality of latent space of the gan")
    parser.add_argument("--image_shape", type=lambda x: map(int, x.split(',')), default=(3, 256, 256),
                        help="Shape of the resulting image")
    parser.add_argument("--lr", type=float, default=0.1,
         help="Learning rate for Gradient Accent, tao for Langevein Dynamycs, transition var for MetropolisHastings")
    parser.add_argument("--lr_decay", type=float, default=1, help="Multiple lr by this on fail")
 

    parser.add_argument("--optimizer", type=str, default='langevin', choices=['grad', 'langevin', 'hamiltonyan', 'mh', 'baseline'])
    parser.add_argument("--content_image", default='cornell_cropped.jpg')
    parser.add_argument("--adaptive_grad", default=0, type=int, help="Divide gradient by moving average square norm."
                                                                     " Can be userfull for adaptive alpha")

    parser.add_argument("--output_dir", default='output/memorability-evaluation-adaptive_alpha')
    parser.add_argument("--display_ratio", type=int, default=1)
    parser.add_argument("--ham_iters", type=int, default=3)
    parser.add_argument("--number_of_iters", type=int, default=20)
    parser.add_argument("--score_type", choices=['blue', 'mem', 'aes'], default='mem',
                        help="Score type 'blue' is making image more blue, 'mem' - make an image more memoreble")
    parser.add_argument("--weight_image", type=float, default=100, help='Weight of the image score')

    parser.add_argument("--content_images_folder", default='/data4/aliaksandr/memorability/lamem/images/', help="Content images for evaluation")
    parser.add_argument("--content_images_names_file", default='dataset/memorability_test_images.txt', help="File with content image names")
    # parser.add_argument("--content_images_folder", default='dataset/tmp', help="Content images for evaluation")
    # parser.add_argument("--content_images_names_file", default='dataset/tmp.txt', help="File with content image names")
    # parser.add_argument("--external_scorer", default='models/mem_external.h5', help='External memorability scorer')
    # parser.add_argument("--internal_scorer", default='models/mem_internal.h5', help="Internal memorability scorer")
    parser.add_argument("--styles_images_dir", default='dataset/devian_art',
                                        help='Directory with styles for baseline evaluation')

    args = parser.parse_args()

    optimizers = {'grad': GradientAccent, 'langevin': LangevinMCMC,
                  'mh': MetropolisHastingsMCMC, 'hamiltonyan': partial(HamiltonyanMCMC, L=args.ham_iters),
                  'baseline': None}
    args.optimizer = optimizers[args.optimizer]

    return args
