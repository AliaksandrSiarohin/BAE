import sqlite3
from argparse import ArgumentParser
import os
from subprocess import Popen
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--number_of_images", default=100, type=int)
parser.add_argument("--result_folder", default=None)
parser.add_argument("--category", default="media_watercolor")

args = parser.parse_args()

if args.result_folder is None:
    args.result_folder = 'dataset/' + args.category

if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

db = sqlite3.connect('dataset/bam.sqlite')
cursor = db.cursor()

cursor.execute('select src from modules, scores where modules.mid = scores.mid order by %s desc limit 0,%s;' %
               (args.category, args.number_of_images))


for i, src in tqdm(enumerate(cursor.fetchall())):
    FNULL = open(os.devnull, 'w')
    name = src[0]
    ext = name[name.rfind('.'):]
    output = Popen(['curl', '-f', src[0], '-o', os.path.join(args.result_folder, str(i) + ext)], stdout=FNULL, stderr=FNULL)
    output.wait()
