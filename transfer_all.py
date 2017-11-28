from style_transfer_model import style_transfer_model, deprocess_input, preprocess_input
from argparse import ArgumentParser
import os
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--content_images_folder", default="../texture_nets/datasets/abstract_art_test")
parser.add_argument("--styles_images_folder", default="../texture_nets/datasets/abstract_art_seeds")
parser.add_argument("--out_images_folder", default="../texture_nets/datasets/abstract_art_adain")

args = parser.parse_args()
model = style_transfer_model()

if not os.path.exists(args.out_images_folder):
    os.makedirs(args.out_images_folder)

for i, style_name in enumerate(os.listdir(args.styles_images_folder)):
    print ("Processing style %i, %s" % (i, style_name))
    out_folder = os.path.join(args.out_images_folder, style_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    style = imread(os.path.join(args.styles_images_folder, style_name))
    style = resize(style, (256, 256), preserve_range=True)
    style = preprocess_input(style)

    for content_name in tqdm(os.listdir(args.content_images_folder)):
        res_file = os.path.join(out_folder, content_name)
        content = imread(os.path.join(args.content_images_folder, content_name))
        content = resize(content, (256, 256), preserve_range=True)
        content = preprocess_input(content)

        res = model.predict([content, style])
        res = deprocess_input(res)
        imsave(res_file, res)
