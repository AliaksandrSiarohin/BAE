from argparse import ArgumentParser
import os
from sklearn.model_selection import train_test_split
import shutil

parser = ArgumentParser()
parser.add_argument("--frac_external", default=0.5, type=float)
parser.add_argument("--folder", default="dataset/emotion_scary")
parser.add_argument("--seed", default=0, type=int)

args = parser.parse_args()

external_folder = os.path.join(args.folder, 'external')
internal_folder = os.path.join(args.folder, 'internal')

if not os.path.exists(external_folder):
    os.makedirs(external_folder)
    os.makedirs(os.path.join(external_folder, 'positive'))
    os.makedirs(os.path.join(external_folder, 'negative'))

if not os.path.exists(internal_folder):
    os.makedirs(internal_folder)
    os.makedirs(os.path.join(internal_folder, 'positive'))
    os.makedirs(os.path.join(internal_folder, 'negative'))


def move_images(subfolder):
    images = os.listdir(os.path.join(args.folder, subfolder))
    external, internal = train_test_split(images, train_size=args.frac_external, random_state=args.seed)
    for img in external:
        shutil.copy(os.path.join(args.folder, subfolder, img), os.path.join(external_folder, subfolder, img))
    for img in internal:
        shutil.copy(os.path.join(args.folder, subfolder, img), os.path.join(internal_folder, subfolder, img))


move_images('positive')
move_images('negative')


