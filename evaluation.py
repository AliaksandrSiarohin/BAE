import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
import os
import pandas as pd
from skimage.transform import resize
from skimage.color import gray2rgb

from cmd import parse_args
from losses import get_image_memorability, get_image_aesthetics, get_image_emotion
from style_transfer_model import preprocess_input, style_transfer_model, deprocess_input, get_encoder
from mixer import ChainMix

import keras.backend as K
from keras.backend import tf as ktf
config = ktf.ConfigProto()
config.gpu_options.allow_growth = True
session = ktf.Session(config=config)
K.set_session(session)


class Scorer(object):
    def __init__(self, score_type):
        assert score_type in ['aes', 'mem', 'scary', 'gloomy', 'happy', 'peaceful']
        if score_type == 'mem':
            external = 'models/mem_external.h5'
            internal = 'models/mem_internal.h5'
            score_fun = get_image_memorability
        elif score_type == 'aes':
            external = 'models/ava2.h5'
            internal = 'models/ava2.h5'
            score_fun = get_image_aesthetics
        else:
            external = 'models/%s_external.h5' % score_type
            internal = 'models/%s_internal.h5' % score_type
            score_fun = get_image_emotion
 
        image = K.placeholder(shape=(1, 3, None, None))
        self.f_external = K.function([image, K.learning_phase()], [score_fun(image=image, net=external)])
        self.f_internal = K.function([image, K.learning_phase()], [score_fun(image=image, net=internal)])

    def compute_external(self, images):
        scores = []
        for image in images:
            img_pr = preprocess_input(image)
            scores.append(np.squeeze(self.f_external([img_pr, 0])[0]))
        return np.array(scores)

    def compute_internal(self, images):
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

    scorer = Scorer(score_type=args.score_type)

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

        initial_memorability = scorer.compute_external([img])[0]

        external_scores = scorer.compute_external(generated_images)
        internal_scores = scorer.compute_internal(generated_images)

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


encoder = None
prediction_model = None
def compute_content_loss(img1_path, img2_path):
    global prediction_model
    from keras.applications import vgg19 as model
    if prediction_model is None:
        prediction_model = model.VGG19()

    img1 = imread(img1_path)
    img1 = gray2rgb(img1) if len(img1.shape) == 2 else img1
    img1 = np.expand_dims(resize(img1, (224, 224), preserve_range=True), 0)
    img1 = model.preprocess_input(img1)

    img2 = imread(img2_path)
    img2 = gray2rgb(img2) if len(img2.shape) == 2 else img2
    img2 = np.expand_dims(resize(img2, (224, 224), preserve_range=True), 0)
    img2 = model.preprocess_input(img2)

    predictions1 = prediction_model.predict(img1)
    predictions2 = prediction_model.predict(img2)

    cross_entropy = -np.sum(predictions1 * np.log(predictions2) + (1 - predictions1) * np.log(1 - predictions2))

    return cross_entropy


def compute_style_descriptor(img_path, encoder_path):
    global encoder
    if encoder is None:
        encoder = get_encoder(encoder_path)

    img = imread(img_path)
    img = preprocess_input(resize(img, (256, 256), preserve_range=True))

    descriptor = encoder.predict(img)
    descriptor = np.squeeze(descriptor, axis=0)

    #print descriptor.shape

    descriptor = descriptor.reshape((512, -1))
    mu = np.mean(descriptor, axis=1)
    sigma = np.log(np.var(descriptor, axis=1) + 1e-5)

    return np.concatenate([mu, sigma])


def compute_style_variance(img_paths, encoder_path):
    descriptos = []
    for img_path in img_paths:
        descriptos.append(compute_style_descriptor(img_path, encoder_path))

    descriptos = np.vstack(descriptos)
    elemntwise_var = np.var(descriptos, axis=0)
    return np.mean(elemntwise_var)


def compute_top_score(df, args, type, top=5, reinit=100000, content_loss=True, style_var=True):
    scores_gaps = []
    scores_style_var = []
    scores_content = []

    encoder_path = args.encoder
    generated_images_path = os.path.join(args.output_dir, type)
    content_images_path = args.content_images_folder

    for content in tqdm(np.unique(df['content_names'])):
        internal_scores = np.array(df[df['content_names'] == content]['internal_scores'])
        gaps = np.array(df[df['content_names'] == content]['gaps'])
        styles = np.array(df[df['content_names'] == content]['style_names'])
        chosen_styles = []

        for i in range(0, len(internal_scores), reinit):
            internal_scores_slice = internal_scores[i:(i+reinit)]
            gaps_slice = gaps[i:(i+reinit)]

            sorted = np.argsort(internal_scores_slice)[::-1]
            scores_gaps.append(np.mean(gaps_slice[sorted][:top]))

            styles_slice = styles[i:(i+reinit)]
            chosen_styles += list(styles_slice[sorted][:top])

        if content_loss:
            for style in chosen_styles:
                score = compute_content_loss(os.path.join(generated_images_path, content[:-4], style),
                                                 os.path.join(content_images_path, content))
                scores_content.append(score)

        if style_var:
            style_paths = [os.path.join(generated_images_path, content[:-4], style) for style in chosen_styles]
            score = compute_style_variance(style_paths, encoder_path)
            scores_style_var.append(score)


    return np.mean(scores_gaps), np.mean(scores_style_var), np.mean(scores_content)


if __name__ == "__main__":
    tops = [1, 5, 10]
    args = parse_args()
    
    if args.optimizer is not None:
        #generate_all_images(args=args, scores_file='chain_scores_dataframe.csv', type='chain')
        df = pd.read_csv(os.path.join(args.output_dir, 'chain_scores_dataframe.csv'))
        for top in tops:
            print ("Generated scores top %s : gap %s, style-var %s, content-loss %s" %
                   ((top,) + compute_top_score(df, args, 'chain', top,  args.reinit)))
    
    if args.optimizer is None:
        #generate_all_images(args=args, scores_file='baseline_scores_dataframe.csv', type='baseline')
        df = pd.read_csv(os.path.join(args.output_dir,'baseline_scores_dataframe.csv'))
        for top in tops:
            print ("Generated scores top %s: gap %s, style-var %s, content-loss %s" %
                    ((top,) + compute_top_score(df, args, 'baseline', top, args.reinit)))
 
