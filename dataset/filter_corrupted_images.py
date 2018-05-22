from skimage.io import imread
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--folder", default="dataset/emotion_scary")

args = parser.parse_args()

for dir_name, _, files in os.walk(args.folder):
    print('Found directory: %s' % dir_name)
    for fname in files:
        img_path = os.path.join(dir_name, fname) 
        try:
           imread(img_path)
        except:
           print img_path
           os.remove(img_path)


