import numpy as np
from scipy.special import logit
from skimage.io import imread, imsave
import os

from evaluation import Scorer
from mixer import ChainMix
from cmd import parse_args


def print_image_score(score, score_type):
    if score_type == 'blue':
        print ("Image blue score %s" % score)
    elif score_type == 'mem':
        score = logit(np.exp(score))
        print ("Image memorability score %s" % score)
    return score


def add_text_to_image(image_path, text):
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, (0,0,0))
    img.save(image_path)


def main():
    args = parse_args()

    img = imread(args.content_image)

    mixer = ChainMix(args)

    mixer.compile()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generated_images, alphas = mixer.run(img, verbose=True)
    print alphas
    if args.score_type == 'mem' or args.score_type == 'aes':
        scorer = Scorer(args.score_type)
        initial_image_score = scorer.compute_external([img])

        print ("Initial image score %s" % (initial_image_score, ))

        scores = scorer.compute_internal(generated_images)
        print scores
        for i, image in enumerate(generated_images):
            img_path = os.path.join(args.output_dir, 'img%s.jpg') % i
            imsave(img_path, image)
            add_text_to_image(img_path, str(scores[i]))
    else:
        for i, image in enumerate(generated_images):
            img_path = os.path.join(args.output_dir, 'img%s.jpg') % i
            imsave(img_path, image)

if __name__ == "__main__":
    main()
