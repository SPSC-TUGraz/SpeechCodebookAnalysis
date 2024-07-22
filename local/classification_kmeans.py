# Author: Julian Linke (linke@tugraz.at)
# SPSC TU Graz (July 2023)

import os, sys
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
import faiss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from itertools import permutations

plt.rcParams.update({'font.size': 15})

def find_best_map(original_map, y_true, spks, distances):
    print("-- (INFO) best map between y_pred and y_true (based on accuracy)")
    keys, values = zip(*original_map.items())
    value_perms = permutations(values)
    best_map = None
    best_accuracy = -np.inf
    for value_perm in value_perms:
        temp_map = dict(zip(keys, value_perm))
        y_pred = []
        col = []
        for idx, (spk, distance) in enumerate(zip(spks, distances)):
            clus_idx = np.argmin(distance)
            y_pred.append(temp_map[clus_idx])
        accuracy = accuracy_score(y_true, y_pred)  
        if accuracy > best_accuracy:
            best_map = temp_map
            best_accuracy = accuracy
    print(f'Best map: {best_map}')
    print(f'Best Accuracy: {best_accuracy}')
    return best_map

def plot_CM(y_true, y_pred, class_names):
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [
    ("confusion matrix without normalization", None),
    ("normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=class_names,
            labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
            colorbar=False,
            values_format='.2f'
        )
        plt.xticks(rotation=90)
        #disp.ax_.set_title(title)
        print(f"-- (INFO) {title}")
        print(disp.confusion_matrix)
    return disp


def main(X_path, y_path, pcaA_path, pcab_path, out_path, json_path, NC):
    # load json ... 
    # dict with {"corpusA_speakingstyle": [spk1 spk2,...], 
    #            "corpusB_speakingstyle": [spk1 spk2,...], ...}
    with open(json_path, 'r') as f:
        corpora = json.load(f)

    X = np.load(X_path)
    spks = np.load(y_path, allow_pickle=True)

    A = np.load(pcaA_path)
    b = np.load(pcab_path)
    print('\nspkIDs:\n{}'.format(spks))
    print(f'\ninput matrix X (shape: {np.shape(X)}):\n{X}')
    print(f'\nprojection matrix A (shape: {np.shape(A)}):\n{A}')
    
    # PROJECTION COMBINATIONS
    proj_del = {2: ['PCA1', 'PCA2'],
                1: ['PCA1', 'PCA3'],
                0: ['PCA2', 'PCA3'],
                'none': ['PCA1', 'PCA2', 'PCA3']}

    y_true = []
    for key, spkIDs in corpora.items():
        count = len(spkIDs)
        y_true.append([key]*count)
    y_true = [item for sublist in y_true for item in sublist]
    
    y_pred_map = {}
    for idx, key in enumerate(np.unique(y_true)):
        y_pred_map[idx] = key

    for pdel in proj_del.keys():
        print("\n----\nProject with PCA dimensions {}\n----".format(proj_del[pdel]))
        X_proj3D = np.dot(X, A) + b # [styles x 320^2] * [320^2 x 2] or [styles x corpora] + [corpora x 2]
        if pdel == 'none':
            X_proj = X_proj3D
        else:
            X_proj = np.delete(X_proj3D, pdel, 1)

        print("... kmeans ...")
        d = X_proj3D.shape[1]
        kmeans = faiss.Kmeans(
            d,
            int(NC),
            niter=100,
            verbose=True,
            gpu=False,
        )
        kmeans.train(X_proj3D)

        if pdel == 'none':
            centroids = kmeans.centroids
        else:
            centroids = np.delete(kmeans.centroids, pdel, 1)
        print('-- (INFO) centroids (3D)\n{}'.format(kmeans.centroids))
        print('-- (INFO) centroids (without dimension {})\n{}'.format(pdel, centroids))
        print('-- (INFO) centroids shape: {}'.format(centroids.shape))

        col, y_pred = [], []
        col_map = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'olive', 6: 'pink'}
        distances = pairwise_distances(X_proj, centroids, metric='euclidean')
        y_pred_map = find_best_map(y_pred_map, y_true, spks, distances)
        print("-- (INFO) resulting distances to k-means centroids")
        for idx, (spk, distance) in enumerate(zip(spks, distances)):
            print(f"{spk}: {distance}")
            clus_idx = np.argmin(distance)
            y_pred.append(y_pred_map[clus_idx])
            col.append(col_map[clus_idx])

        fig, ax = plt.subplots(figsize=(30,30))
        print('-- (INFO) y_true:\n{}\n-- (INFO) y_pred:\n{}'.format(y_true,y_pred))
        disp = plot_CM(y_true, y_pred, list(y_pred_map.values()))
        ax = disp.ax_
        for text in ax.texts:
            value = float(text.get_text())
            if value == 0.00:
                text.set_text('0')
            else:
                text.set_text('{:.2f}'.format(value))
        plt.tight_layout()
        PCAlabel = '_'.join([s for s in proj_del[pdel]])
        title='CM_{}_{}_nclus{}.png'.format(X_path.split('/')[-1].replace('.npy',''), PCAlabel, NC)
        plt.savefig(os.path.join(out_path,title))

        print("-- (INFO) summary scores")
        print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
        print('Precision-score: {}'.format(precision_score(y_true, y_pred, average='weighted')))
        print('Recall-score: {}'.format(recall_score(y_true, y_pred, average='weighted')))
        print('F1-score: {}'.format(f1_score(y_true, y_pred, average='weighted')))

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
    ############################# PCA A PATH ###########################
    try:
        pcaA_path = sys.argv[3]
        print("pca matrix A path is: " + pcaA_path)
    except:
        print("ERROR: pcaA_path not specified")
    ############################# PCA b PATH ###########################
    try:
        pcab_path = sys.argv[4]
        print("pca vector b path is: " + pcab_path)
    except:
        print("ERROR: pcab_path not specified")
    ############################# OUT PATH ###########################
    try:
        out_path = sys.argv[5]
        print("output path is: " + out_path)
    except:
        print("ERROR: out_path not specified")
    ############################# JSON PATH ###########################
    try:
        json_path = sys.argv[6]
        print("json path is: " + json_path)
    except:
        print("ERROR: json_path not specified")
    ############################# NUMBER CLUSTERS ###########################
    try:
        NC = sys.argv[7]
        print("NC (Number Clusters) is: " + NC)
    except:
        print("ERROR: NC (Number Clusters) not specified")

    main(X_path, y_path, pcaA_path, pcab_path, out_path, json_path, NC)