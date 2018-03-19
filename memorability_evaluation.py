import keras.backend as K

import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from mixer import ChainMix
import os
import pandas as pd
from cmd import parse_args
from skimage.transform import resize

from losses import get_image_memorability
from style_transfer_model import preprocess_input, style_transfer_model, deprocess_input

#import gan.train

import keras.backend as K
#assert K.image_data_format() == 'channels_last', "Backend should be tensorflow and data_format channel_last"
from keras.backend import tf as ktf
config = ktf.ConfigProto()
config.gpu_options.allow_growth = True
session = ktf.Session(config=config)
K.set_session(session)


class MemorabilityScorer(object):
    def __init__(self, mem_external='models/mem_external.h5', mem_internal='models/mem_internal.h5'):
        image = K.placeholder(shape=(1, 3, None, None))
        self.f_external = K.function([image, K.learning_phase()], [get_image_memorability(image=image, memnet=mem_external)])
        self.f_internal = K.function([image, K.learning_phase()], [get_image_memorability(image=image, memnet=mem_internal)])

    def compute_memorability_external(self, images):
        scores = []
        for image in images:
            img_pr = preprocess_input(image)
            scores.append(np.squeeze(self.f_external([img_pr, 0])[0]))
        return np.array(scores)

    def compute_memorability_internal(self, images):
        scores = []
        for image in images:
            img_pr = preprocess_input(image)
            scores.append(np.squeeze(self.f_internal([img_pr, 0])[0]))
        return np.array(scores)


def save_images(out_folder, generated_images, names):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for i, image in enumerate(generated_images):
        imsave(os.path.join(out_folder, names[i]), image)


def scores_to_df(initial_memorability, external_scores, internal_scores, style_names, content_names, alphas):
    d = {'external_scores': external_scores,
         'internal_scores': internal_scores,
         'style_names': style_names,
         'content_names': content_names,
         'gaps': external_scores - initial_memorability,
         'alpha': alphas}
    return pd.DataFrame(data=d)


def chain_generation(img, mixer, seed):
    generated_images, alphas = mixer.run(img, verbose=False, seed=seed)
    style_names = map(str, range(len(generated_images)))
    style_names = map(lambda x: x + '.jpg', style_names)

    return generated_images, style_names, alphas


def baseline_generation(img, model, styles_images_dir, alpha):
    img_content = resize(img, (256, 256), preserve_range=True)
    img_content = preprocess_input(img_content)

    generated_images = []
    style_names = os.listdir(styles_images_dir)
    for style_name in style_names:
        img_style = imread(os.path.join(styles_images_dir, style_name))
        img_style = resize(img_style, (256, 256), preserve_range=True)
        img_style = preprocess_input(img_style)

        res = model.predict([img_content, img_style])
        generated_images.append(deprocess_input(res))
    return generated_images, style_names, alpha * np.ones(len(generated_images))


def generate_all_images(args, scores_file, type):
    assert type in ['baseline', 'chain']

    content_images_folder = args.content_images_folder
    content_images_names_file = args.content_images_names_file

    with open(content_images_names_file) as f:
        content_images_names = f.read().split('\n')

    mem_scorer = MemorabilityScorer(args.external_scorer, args.internal_scorer)

    if type == 'chain':
        mixer = ChainMix(args)
        mixer.compile()
    else:
        model = style_transfer_model(alpha=args.alpha_mean,
                                     decoder=args.decoder,
                                     encoder=args.encoder)

    for i, content_image in tqdm(list(enumerate(content_images_names))):
        if content_image.strip() == '':
            continue

        out_dir = os.path.join(args.output_dir, os.path.join(type, content_image[:-4]))
        if os.path.exists(out_dir):
            print ("%s exists. Skip this content image." % out_dir)
            continue

        img = imread(os.path.join(content_images_folder, content_image))

        if type == 'chain':
            generated_images, style_names, alphas = chain_generation(img, mixer, seed=i)
        else:
            generated_images, style_names, alphas = baseline_generation(img, model, args.styles_images_dir, args.alpha_mean)

        initial_memorability = mem_scorer.compute_memorability_external([img])[0]

        external_scores = mem_scorer.compute_memorability_external(generated_images)
        internal_scores = mem_scorer.compute_memorability_internal(generated_images)


        content_names = np.repeat(content_image, repeats=len(external_scores))

        df = scores_to_df(initial_memorability, external_scores, internal_scores, style_names, content_names, alphas)
        save_images(out_dir, generated_images, style_names)

        f_name = os.path.join(args.output_dir, scores_file)
        if not os.path.exists(os.path.join(f_name)):
            with open(f_name, 'a') as f:
                df.to_csv(f, index=False, header=True)
        else:
            with open(f_name, 'a') as f:
                df.to_csv(f, index=False, header=False)

def compute_top_score(df, top=5):
    scores = []
    for content in np.unique(df['content_names']):
        external_scores = np.array(df[df['content_names'] == content]['external_scores'])
        gaps = np.array(df[df['content_names'] == content]['gaps'])
        sorted = np.argsort(external_scores)[::-1]
        scores.append(np.mean(gaps[sorted][:top]))
    return np.mean(scores)

if __name__ == "__main__":
    tops = [1, 5, 10]
    args = parse_args()
    generate_all_images(args=args, scores_file='chain_scores_dataframe.csv', type='chain')
    df = pd.read_csv(os.path.join(args.output_dir, 'chain_scores_dataframe.csv'))
    for top in tops:
        print ("Generated scores top %s: %s" % (top, compute_top_score(df, top)))

    generate_all_images(args=args, scores_file='baseline_scores_dataframe.csv', type='baseline')
    df = pd.read_csv(os.path.join(args.output_dir,'baseline_scores_dataframe.csv'))
    for top in tops:
        print ("Baseline scores top %s: %s" % (top, compute_top_score(df, top)))


