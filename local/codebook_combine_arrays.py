# Author: Julian Linke (linke@tugraz.at)
# SPSC TU Graz (July 2023)

import os, sys
import fairseq
import torch, torchaudio
import numpy as np
import json

def main(exp_path, json_path):
    # load json ... 
    # dict with {"corpusA_speakingstyle": [spk1 spk2,...], 
    #            "corpusB_speakingstyle": [spk1 spk2,...], ...}
    with open(json_path, 'r') as f:
        corpora = json.load(f)

    cnt = 0
    for corpus in corpora:
        freqN_path = os.path.join(exp_path, corpus)
        for freqN_vec in os.listdir(freqN_path):
            if 'freqN' in freqN_vec:
                cnt = cnt+1
    print('counted {} freqN-files'.format(cnt))

    freqN_convs = np.zeros((cnt, 320**2), dtype=np.float32)
    spks_vec = np.zeros((cnt,), dtype=object) 
    idx = 0
    print('... start combining freqN_*.npy-files coming from path {}'.format(exp_path))
    for corpus in corpora:
        style = corpus.split("_")[1] # second entry is always style
        freqN_path = os.path.join(exp_path, corpus)
        for freqN_vec in os.listdir(freqN_path):
            if 'freqN' in freqN_vec:
                #print(freqN_vec)
                split = freqN_vec.split('_')[1].replace('.npy','')
                print('read and append split {} ...'.format(os.path.join(freqN_path,freqN_vec)))
                # combine
                freqN_convs[idx,:] = np.load(os.path.join(freqN_path,freqN_vec))
                spks_vec[idx] = f'{split}{style}'
                idx = idx + 1
    freq_path, splits_path = os.path.join(exp_path, 'splits_freqs'), os.path.join(exp_path, 'splits_labels')
    print('\nwrite {}.npy and {}.npy'.format(freq_path, splits_path))
    np.save(freq_path, freqN_convs)
    np.save(splits_path, spks_vec)
    print('\nwrite {}.tsv and {}.tsv'.format(freq_path, splits_path))
    np.savetxt(freq_path+'.tsv', freqN_convs, delimiter='\t')
    np.savetxt(splits_path+'.tsv', spks_vec, fmt='%s', delimiter='\t')
    
if __name__ == "__main__":
    ############################# EXP PATH ###########################
    try:
        exp_path = sys.argv[1]
        print("exp path is: " + exp_path)
    except:
        print("ERROR: data_path not specified")
    ############################# JSON PATH ###########################
    try:
        json_path = sys.argv[2]
        print("json path is: " + json_path)
    except:
        print("ERROR: json_path not specified")

    main(exp_path, json_path)
