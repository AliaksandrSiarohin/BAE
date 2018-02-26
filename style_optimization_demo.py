import numpy as np
from scipy.special import logit
from skimage.io import imread, imsave
import os

from memorability_evaluation import MemorabilityScorer
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
    if not os.path.exists(args.samples_dir):
        os.makedirs(args.samples_dir)

    generated_images = mixer.run(img, verbose=True)

    if args.score_type == 'mem':
        mem_scorer = MemorabilityScorer()
        initial_image_score = mem_scorer.compute_memorability_external([img])
        print ("Initial image memorability %s" % (initial_image_score, ))

        scores = mem_scorer.compute_memorability_internal(generated_images)
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
