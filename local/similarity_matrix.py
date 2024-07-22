# Author: Julian Linke (linke@tugraz.at)
# SPSC TU Graz (July 2023)

import os, sys
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 

from scipy.stats import entropy
from numpy.linalg import norm

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 1 - (0.5 * (entropy(_P, _M) + entropy(_Q, _M)))

def JSD_similarity(X):
    sim = np.zeros((np.shape(X)[0], np.shape(X)[0]), dtype=np.float32)
    print('\ncalculate JSD similarity matrix of X: {} ...'.format(X.shape))
    for row, x in enumerate(X):
        for col, _ in enumerate(sim):
            #print('calculate JSD similarity of features ({}, {})'.format(row, col))
            sim[row, col] = JSD(x, X[col,:])
    return sim

def main(X_path, y_path, out_path, json_path):
    # load json ... 
    # dict with {"corpusA_speakingstyle": [spk1 spk2,...], 
    #            "corpusB_speakingstyle": [spk1 spk2,...], ...}
    with open(json_path, 'r') as f:
        corpora = json.load(f)

    # LOAD MATRIX AND SPLITS
    X = np.load(X_path)
    splits = np.load(y_path, allow_pickle=True)
    print('\ninput splits: {}\n... len: {}'.format(splits, len(splits)))
    corpora_lengths = {key: len(value) for key, value in corpora.items()}
        
    # SORT:
    d_splits, col = {}, []
    for idx, split in enumerate(splits):
        d_splits[split] = X[idx,:]
    X = np.zeros((np.shape(X)), dtype=np.float32)
    splits = np.zeros((np.shape(splits)), dtype=object)
    i = 0
    for corpus in corpora:
        for split in corpora[corpus]:
            style = corpus.split("_")[1] # second entry is always style
            split = f'{split}{style}'
            if split in d_splits:
                X[i,:] = d_splits[split]
                splits[i] = split
                i = i + 1
            else: print('Wrong corpora entry: {}?'.format(split))
    print('\nsorted splits: {}\n... len: {}'.format(splits, len(splits)))

    print('\nwrite sorted {} and {}'.format(X_path, y_path))
    np.save(X_path, X)
    np.save(y_path, splits)
    print('write sorted {} and {}'.format(X_path, y_path))
    np.savetxt(X_path.replace('.npy','.tsv'), X, delimiter='\t')
    np.savetxt(y_path.replace('.npy','.tsv'), splits, fmt='%s', delimiter='\t')

    # Calculate similarity matrix and save:
    X_sim = JSD_similarity(X)
    np.save(os.path.join(out_path,'similarity_matrix'), X_sim)
    np.savetxt(os.path.join(out_path,'similarity_matrix.tsv'), X_sim, delimiter='\t')

if __name__ == "__main__":
    ############################# FEATURES PATH ###########################
    try:
        X_path = sys.argv[1]
        print("input matrix path is: " + X_path)
    except:
        print("ERROR: X_path not specified")
    ############################# LABELS PATH ###########################
    try:
        y_path = sys.argv[2]
        print("label vector path is: " + y_path)
    except:
        print("ERROR: y_path not specified")
    ############################# OUT PATH ###########################
    try:
        out_path = sys.argv[3]
        print("output path is: " + out_path)
    except:
        print("ERROR: out_path not specified")
    ############################# JSON PATH ###########################
    try:
        json_path = sys.argv[4]
        print("json path is: " + json_path)
    except:
        print("ERROR: json_path not specified")

    main(X_path, y_path, out_path, json_path)
