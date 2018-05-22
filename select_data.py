import pandas as pd
import os
import numpy as np
import shutil

def select_top(df, use_internal = True, content_to_select = 10):
    pairs = []
    scores = []
    field = 'internal_scores' if use_internal else 'external_scores'
    content_names =  np.unique(df['content_names'])
    np.random.seed(0)
    content_names = np.random.choice(content_names, size=content_to_select, replace=False)
    for content in content_names:
        internal_scores = np.array(df[df['content_names'] == content][field])
        styles = np.array(df[df['content_names'] == content]['style_names'])
        gaps = np.array(df[df['content_names'] == content]['gaps'])
        at_max = np.argmax(internal_scores)
        pairs.append((content, styles[at_max]))
        scores.append(gaps[at_max])
    print (np.mean(scores))
    print (pairs)
    return pairs

def copy_data(in_folder, out_folder, pairs):
   if not os.path.exists(out_folder):
       os.makedirs(out_folder) 
   for pair in pairs:
       shutil.copy(os.path.join(in_folder, pair[0].replace('.jpg', ''), pair[1]), os.path.join(out_folder, pair[0])) 


if __name__ == "__main__":
    df = pd.read_csv(os.path.join('output/scary_evaluation-adaptive_alpha_N05025/chain_scores_dataframe.csv'))
    pairs = select_top(df)
    copy_data("output/scary_evaluation-adaptive_alpha_N05025/chain", 'method', pairs)
    df = pd.read_csv(os.path.join('output/scary_evaluation-fixed_alpha0.5/baseline_scores_dataframe.csv'))
    pairs = select_top(df)
    copy_data("output/scary_evaluation-fixed_alpha0.5/baseline", 'baseline', pairs)

