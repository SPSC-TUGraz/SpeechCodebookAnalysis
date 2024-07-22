# Author: Julian Linke (linke@tugraz.at)
# SPSC TU Graz (July 2023)

import os, sys
import fairseq
import torch, torchaudio
import numpy as np
import json

def print_list(l):
    for s in l:
        print(s)

def count_freq(codebook_all_indexes, freq):
    seq = []
    for t in range(0, len(codebook_all_indexes)):
        codebook_idx_at_time_t = codebook_all_indexes[t].item() # tensor(integer).item()
        if VERBOSE: 
            print('(Verbose) frame={}: used codebook entry: {}'.format(t+1, codebook_idx_at_time_t))
        seq.append(codebook_idx_at_time_t)
        freq[codebook_idx_at_time_t] = freq[codebook_idx_at_time_t] + 1  
    return freq, seq

def calc_codebook_indexes(audio_path, freq, N):
    x, fs = torchaudio.load(audio_path)
    x = x.to(device) # torch.Size([1, 57120]) [1 x Samples]

    if np.shape(x)[1] > 512:
        C = model.quantize(x)
        quantized_features = C[0][0] # torch.Size([178, 768]) [T x d]
        codebook_G2_indices = C[1] # torch.Size([1, 178, 2]) [1 x T x G]; G=2
        codebook_all_indexes = model.quantizer.to_codebook_index(codebook_G2_indices)[0] # torch.Size([178]) [T]
        Nwav = len(codebook_all_indexes)
        freq, seq = count_freq(codebook_all_indexes, freq)
        N = N + Nwav
        print('feature vectors: {}/{} (file/all)'.format(Nwav, N))
    else:
        Nwav = 0
        print('WARNING: Input size of file is {} (smaller than Kernel size), skip ...'.format(len(x)))
    return freq, Nwav

def main(exp_path, lst_path, json_path, model_path, VERBOSE):
    # set model and device global
    global model, device
    #device = torch.device('cpu')
    device = torch.device('cuda')
    # load existing model
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = model[0]
    model = model.to(device)
    # load json ... 
    # dict with {"corpusA_speakingstyle": [spk1 spk2,...], 
    #            "corpusB_speakingstyle": [spk1 spk2,...], ...}
    with open(json_path, 'r') as f:
        corpora = json.load(f)
    # corpus loop
    for corpus in corpora.keys():
        # LOGFILE:
        if VERBOSE: 
            sys.stdout = open(os.path.join(exp_path, 'logs', "codebook_freqs_{}.log".format(corpus)),"w")
        # speaker list and style
        spks = corpora[corpus]
        style = corpus.split("_")[1] # second entry is always style
        # prepare numpy arrays
        freqN_spks = np.zeros((len(spks), 320**2), dtype=np.float32) # [SPKS x 120400]
        spks_vec = np.zeros((len(spks),), dtype=object) # [SPKS x 1]
        # speaker loop
        for idx, spk in enumerate(spks):
            print(f"\n--- speaker {spk} in corpus {corpus} ---")
            freq, N = dict.fromkeys(range(1, 320**2+1), 0), 0
            processed_files = []
            # extract frequencies per speaker
            with open(lst_path, 'r') as tsv:
                rows = tsv.readlines()
                for row in rows:
                    uttID, audio_path = row.split()
                    corpus_match = ('_').join(audio_path.split('/')[2].split('_')[1:]) # DATA/expname/data_corpus_speakingstyle/spk/*wav
                    spk_match = audio_path.split('/')[3] # DATA/expname/data_corpus_speakingstyle/spk/*wav
                    if corpus == corpus_match and spk == spk_match: 
                        print('\nread wav-file {}'.format(audio_path))
                        freq, Nwav = calc_codebook_indexes(audio_path, freq, N)
                        if Nwav != 0:
                            N = N + Nwav
                            processed_files.append(audio_path)
                print(f'(DONE) Found {N} observations for speaker {spk} ...')
            # combine frequencies
            freqN_vec = np.zeros((1, 320**2), dtype=np.float32)
            os.system('mkdir -p {}'.format(os.path.join(exp_path, 'txt', corpus)))
            os.system('mkdir -p {}'.format(os.path.join(exp_path, 'numpy', corpus)))
            with open(os.path.join(exp_path, 'txt', corpus, 'freq_{}.txt'.format(spk)), 'w') as ffreq, \
                open(os.path.join(exp_path, 'txt', corpus, 'freqN_{}.txt'.format(spk)), 'w') as ffreqN:
                for i, code_entry in enumerate(freq.keys()):
                    ffreq.write('{}\t{}\n'.format(code_entry, freq[code_entry]))
                    ffreqN.write('{}\t{}\n'.format(code_entry, freq[code_entry]/N))
                    freqN_vec[0,i] = freq[code_entry]/N
            np.save(os.path.join(exp_path, 'numpy', corpus, 'freqN_{}'.format(spk,style)), freqN_vec)
            # combine frequencies per corpus
            print(f"... speaker {spk} is column {idx} of array freq_{corpus}.npy!")
            freqN_spks[idx,:] = freqN_vec
            spks_vec[idx] = f'{spk}{style}'
        # write combined frequencies
        np.save(os.path.join(exp_path, 'numpy', corpus, 'freq_{}'.format(corpus)), freqN_spks)
        np.save(os.path.join(exp_path, 'numpy', corpus, 'spkIDs_{}'.format(corpus)), spks_vec)

if __name__ == "__main__":
    ############################# EXP PATH ###########################
    try:
        exp_path = sys.argv[1]
        print("\nexp path is: " + exp_path)
    except:
        print("ERROR: data_path not specified")
    ############################# LIST PATH ###########################
    try:
        lst_path = sys.argv[2]
        print("list path is: " + lst_path)
    except:
        print("ERROR: lst_path not specified")
    ############################# JSON PATH ###########################
    try:
        json_path = sys.argv[3]
        print("json path is: " + json_path)
    except:
        print("ERROR: json_path not specified")
    ############################# MODEL PATH ###########################
    try:
        model_path = sys.argv[4]
        print("model path is: " + model_path)
    except:
        print("ERROR: model_path not specified")
    ############################# VERBOSE #############################
    global VERBOSE
    try:
        VERBOSE = int(sys.argv[5])
        print("VERBOSE is " + str(VERBOSE) + "\n")
    except:
        print("VERBOSE is not specified, default is 0!")
        VERBOSE = 0

    main(exp_path, lst_path, json_path, model_path, VERBOSE)
