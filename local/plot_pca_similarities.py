# Author: Julian Linke (linke@tugraz.at)
# SPSC TU Graz (July 2023)

import os, sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
import faiss
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# List of predefined colors for each corpus (you can add more if necessary)
colors = ['tab:red', 'tab:blue', 
          'tab:green', 'tab:orange', 
          'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 
          'tab:olive', 'tab:cyan']

# fontsize and markersize
MS = 30

def main(X_path, y_path, pcaA_path, pcab_path, out_path, json_path):
    # load json ... 
    # dict with {"corpusA_speakingstyle": [spk1 spk2,...], 
    #            "corpusB_speakingstyle": [spk1 spk2,...], ...}
    with open(json_path, 'r') as f:
        corpora = json.load(f)

    smID = X_path.split('/')[-1].replace('.npy','')
    X = np.load(X_path)
    spks = np.load(y_path, allow_pickle=True)

    A = np.load(pcaA_path)
    b = np.load(pcab_path)
    print('\nspkIDs:\n{}'.format(spks))
    print('\ninput matrix X:\n{}'.format(X))

    labels = list(spks)
    fig, ax = plt.subplots(figsize=(15,15))
    cax = ax.matshow(X, interpolation='nearest')
    # x-ticks fontsize
    for label in ax.get_xticklabels():
        label.set_fontsize(25)
    # y-ticks fontsize
    for label in ax.get_yticklabels():
        label.set_fontsize(25)
    ax.grid()
    plt.xticks(range(len(labels)), labels, rotation=90);
    plt.yticks(range(len(labels)), labels);
    cbar = fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .8, .9, 1])
    cbar.ax.tick_params(labelsize=25)
    fig_path = os.path.join(out_path,f'{smID}.png')
    print('\nsave similarity matrix to {}'.format(fig_path))
    plt.savefig(fig_path)

    colordict = {}
    for corpus in corpora:
        colordict[corpus] = []
    assert len(corpora.keys()) <= len(colors), 'Not enough colors for corpora'
    for idx, corpus in enumerate(corpora):
        spks_tmp = corpora[corpus]
        for spk in spks_tmp:
            colordict[corpus].append(colors[idx])
    col = sum([lst for lst in colordict.values()], [])

    print("\nProject with PCA (3 dimensions) ...")
    X_proj = np.dot(X, A) + b # [styles x 320^2] * [320^2 x 2] or [styles x corpora] + [corpora x 2]
    print(f"\nshape(X_proj) = {np.shape(X_proj)}")
    print('mean(X_proj,0): {}'.format(np.mean(X_proj,0)))
    
    # 3D plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_proj[:,0], X_proj[:,1], X_proj[:,2], c=col)
    # Set the labels and title
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    # Create the legend
    for k,v in colordict.items():
        ax.scatter([], [], [], c=v[0], label=k, alpha=1, s=MS)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.175, .5))
    # Save the plot
    fig_path = os.path.join(out_path,f'scatter_{smID}_PCA1_PCA2_PCA3.png')
    print('\nsave 3D scatter plot {}'.format(fig_path))
    plt.savefig(fig_path)
    
    plt.rcParams.update({'font.size': 50})
    for proj in [(0, 1), (0, 2), (1, 2)]:
        print(f"\nProject on dimensions {proj}")
        x, y = X_proj[:,proj[0]], X_proj[:,proj[1]]
        fig, ax = plt.subplots(figsize=(18,16))

        for i, txt in enumerate(spks):
            print(f"speaker {txt}: x = {x[i]}, y = {y[i]}")
            ax.plot(x[i], y[i], 'o', color=col[i], markersize=MS, alpha=.5, label=col[i])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        cols = list(by_label.keys())
        for old, new in zip(cols, corpora.keys()):
            by_label[new] = by_label.pop(old)
        ax.legend(by_label.values(), by_label.keys(), fontsize=30)
        plt.xlabel(f'PCA{str(proj[0]+1)}')
        plt.ylabel(f'PCA{str(proj[1]+1)}')
        plt.grid()
        plt.tight_layout()
        pcaID1, pcaID2 = str(proj[0]+1), str(proj[1]+1)
        fig_path = os.path.join(out_path,f'scatter_{smID}_PCA{pcaID1}_PCA{pcaID2}.png')
        print('save 2D scatter plot {}'.format(fig_path))
        plt.savefig(fig_path)

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

    main(X_path, y_path, pcaA_path, pcab_path, out_path, json_path)
