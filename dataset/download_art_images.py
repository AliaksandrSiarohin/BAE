import sqlite3
from argparse import ArgumentParser
import os
from subprocess import Popen
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--number_of_images", default=100000, type=int)
parser.add_argument("--result_folder", default=None)
parser.add_argument("--category", default="emotion_scary")

args = parser.parse_args()

if args.result_folder is None:
    args.result_folder = 'dataset/' + args.category

if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

db = sqlite3.connect('dataset/bam2.sqlite')

cursor = db.cursor()

cursor.execute("select src, label from modules, crowd_labels where modules.mid = crowd_labels.mid and attribute = '%s'"
               "and (label = 'positive' or label = 'negative') limit 0, %s;" % (args.category, args.number_of_images))


for i, src in tqdm(enumerate(cursor.fetchall())):
    FNULL = open(os.devnull, 'w')
    name = src[0]
    label = src[1]
    ext = name[name.rfind('.'):]
    new_name = str(i) + ext

    if not os.path.exists(os.path.join(args.result_folder, label)):
        print os.path.join(args.result_folder, label)
        os.makedirs(os.path.join(args.result_folder, label))
    output = Popen(['curl', '-f', name, '-o', os.path.join(args.result_folder, label, new_name)],
                   stdout=FNULL, stderr=FNULL)
    output.wait()
