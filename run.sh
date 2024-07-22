#!/bin/bash
#set -x

# Author: Julian Linke (linke@tugraz.at)
# SPSC TU Graz (July 2023)

set -e -o pipefail
. path.sh
. conda.sh

if [[ $# -eq 2 ]] ; then
    echo 'ERROR: this run-script requires two arguments: expdata=? expname=? stage=?'
    exit 1
fi

## expname/STAGE
expdata=$1
expname=$2
stage=$3
VERBOSE=1 # write logs for codebook frequency extraction?
printf "\n### STAGE ###\n"
printf "stage: %d\n" $stage
printf "### STAGE ###\n"

## DIRS/PATHS 
model_path=model/xlsr_53_56k_new.pt
exp_dir=exp_$expname

## STAGE 0: DELETE AND RUN ALL STAGES
if [ $stage == 0 ]; then
    printf "\n... Delete old experiment and run all ...\n"
    rm -rf ${exp_dir}
fi

## print:
printf "\nCWD: %s" "$CWD"
printf "\nFAIRSEQ: %s" "$FAIRSEQ"
printf "\nexpname: %s" "$expname"
printf "\nexpdata: %s" "$expdata"
printf "\nmodel_path: %s" "$model_path"
printf "\nexp_dir: %s\n\n" "$exp_dir"

## CREATE EXPERIMENT FOLDER
mkdir -p $exp_dir
mkdir -p $exp_dir/logs
mkdir -p $exp_dir/data
mkdir -p $exp_dir/plots
mkdir -p $exp_dir/txt
mkdir -p $exp_dir/numpy
mkdir -p $exp_dir/numpy/pca

## PREPARE DATA
if [ $stage == 1 ]  || [ $stage == 0 ]; then
    printf "\n... Prepare data (*lst and *json) ...\n"
    python3 local/prepare_data.py --output_lst_path $exp_dir/data/${expname}.lst \
                                 --output_json_path $exp_dir/data/${expname}.json \
                                 --DATA_dir ${expdata}
fi

## COUNT CODEBOOK USAGE
if [ $stage == 2 ] || [ $stage == 0 ]; then
    printf "\n... Count frequencies of codebooks ...\n"
    python3 local/codebook_freqs.py $exp_dir \
                                   $exp_dir/data/${expname}.lst \
                                   $exp_dir/data/${expname}.json \
                                   $model_path \
                                   $VERBOSE
    printf "\n... Combine Arrays ... \n"
    python3 local/codebook_combine_arrays.py $exp_dir/numpy \
                                             $exp_dir/data/${expname}.json
fi

## CALCULATE SIMILARITY MATRIX
if [ $stage == 3 ] || [ $stage == 0 ]; then
    printf "\n... Similarity Matrix ...\n"
    python3 local/similarity_matrix.py \
            $exp_dir/numpy/splits_freqs.npy \
            $exp_dir/numpy/splits_labels.npy \
            $exp_dir/numpy \
            $exp_dir/data/${expname}.json
fi

## PCA SPACE AND PLOTS
if [ $stage == 4 ] || [ $stage == 0 ]; then
    printf "\n... PCA of similarity matrix ...\n"
    python3 $FAIRSEQ/examples/wav2vec/unsupervised/scripts/pca.py \
        $exp_dir/numpy/similarity_matrix.npy \
        --output $exp_dir/numpy/pca \
        --dim 3
    printf "\n... PLOT similarity in PCA space (Analysis) ...\n"
    mkdir -p $exp_dir/plots/analysis
    python3 local/plot_pca_similarities.py \
        $exp_dir/numpy/similarity_matrix.npy \
        $exp_dir/numpy/splits_labels.npy \
        $exp_dir/numpy/pca/3_pca_A.npy \
        $exp_dir/numpy/pca/3_pca_b.npy \
        $exp_dir/plots/analysis \
        $exp_dir/data/${expname}.json
fi

if [ $stage == 5 ] || [ $stage == 0 ]; then
    printf "\n... Classification with k-means in PCA space ...\n"
    mkdir -p $exp_dir/plots/kmeans
    nclust=3 # number of clusters?
    python3 local/classification_kmeans.py \
        $exp_dir/numpy/similarity_matrix.npy \
        $exp_dir/numpy/splits_labels.npy \
        $exp_dir/numpy/pca/3_pca_A.npy \
        $exp_dir/numpy/pca/3_pca_b.npy \
        $exp_dir/plots/kmeans \
        $exp_dir/data/${expname}.json \
        $nclust
fi