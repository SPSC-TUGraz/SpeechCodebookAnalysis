# SpeechCodebookAnalysis

This project contains the code related to the analytical section of our research paper, **"What do self-supervised speech representations encode? An analysis of languages, varieties, speaking styles and speakers"**, which has been accepted for **Interspeech 2023 in Dublin**.

Stay tuned for updates and we appreciate your interest in our work. Please continue exploring this README for more details on the project setup, codebase, and how to navigate through it.

## Dependencies

- Python (version 3.8)
- fairseq (version 0.12.2)
- matplotlib (version 3.7.2)
- scikit-learn (version 1.3.0)
- faiss-cpu (version 1.7.4)

## Repository structure
The repository includes a **main script** (```run.sh```), a folder which includes Python scripts (```local/*py```) and an example data folder (```DATA/GR/```). **If you want to work with your own data you would need to prepare a folder on your own which follows a specific folder structure**. 

The example data folder includes example files from the GRASS corpus (Austrian German) which makes it possible to run an experiment from scratch. In general, the speech data to be analyzed should be stored in the folder ```DATA/```. In case of the example experiment, this folder (```DATA/GR/```) has the following structure:

- ```DATA/GR/data_GR_CS```
  - Various speaker (spkID1, spkID2, ...) folders
    - Various .wav or .flac files (fs=16kHz)
- ```DATA/GR/data_GR_RS```
  - Various speaker (spkID1, spkID2, ...) folders
    - Various .wav or .flac files (fs=16kHz)

The example folder ```GR``` (which must be placed in ```DATA/```) sort of defines one experiment and includes the subfolders ```data_GR_CS``` (GRASS Conversational Speech) and ```data_GR_RS``` (GRASS Read Speech). **Please make sure that those folders are named like this: ```data_{corpus}_{speakingstyle}```**. The audio files should have a sampling rate of 16kHz and can be .wav or .flac files. Given this structure and after installing/preparing all dependencies (see below) you should be able to run the experiment. 

To run a specific stage of the script for a specific dataset, provide the directory where all your data is stored (here ```DATA/GR/```), an experiment name (here ```GR```) and an integer as an argument to the `./run.sh` command. For instance, to run stage ```3``` for the example dataset ```DATA/GR/``` with the experiment name ```GR```, you would use the following command:

```
./run.sh DATA/GR/ GR 3
```

The command automatically generates the experiment folder ```exp_GR```. **Note that stage ```0``` deletes this entire experiment folder (if it exists) and restarts the entire experiment** by running all stages in a row (see below an overview of the stages).

## Reproduction
The following steps are necessary to reproduce the experiment. At first you need to create a conda envrionment and install the necessary packages. Second you have to  clone the fairseq repository and modify the file ```path.sh``` to export necessary environment variables. 

### Conda environment (as of 27th July, 2023)
At first you should create your conda environment:

```
conda create -n speechcodebookanalysis python=3.8
conda activate speechcodebookanalysis
```

To set up your environment, please ensure you have the following Python packages installed. You can install them using pip:

```
pip install fairseq
pip install matplotlib
pip install scikit-learn
pip install faiss-cpu
```

When the environment is created also generate the file ```conda.sh``` which could look like this:

```
source */anaconda3/etc/profile.d/conda.sh
conda activate speechcodebookanalysis
```

The file ```conda.sh``` is sourced at the beginning of ```run.sh```.

### Fairseq Repository (as of 27th July, 2023)
You need to clone the fairseq repository to another directory (e.g., ```../fairseq```).

```
git clone https://github.com/facebookresearch/fairseq.git
``` 

Make sure to modify the file ```path.sh``` in order to export the necessary environment variables. The file ```path.sh``` is also sourced at the beginning of ```run.sh```.

### Model File (as of 27th July, 2023)
**You need to download and store a model file**. In the main script (```run.sh```) you can specify the ```model_path```. This study is based on the large pretrained model **XLSR-53** which can be downloaded here: [wav2vec2](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec)

**Unfortunately loading/initializing the model with version ```fairseq 0.12.2``` lead to errors because of mismatches with respect to dictionary keys. Anyway, we provide a script (```local/create_xlsr_new.py```) which removes some dictionary keys and stores a new version of the model preventing those errors** (see also [ISSUE](https://github.com/facebookresearch/fairseq/issues/3741)).

### Stages

Here is a short overview of the stages of the main script ```run.sh```:

- ```stage=0```: deletes experiment folder (if it exists) and runs all subsequent stages in a row
- ```stage=1```: 
  - prepares the data given an experiment folder (e.g., ```DATA/GR/```)
  - resulting files are stored in ```exp_*/data/```
- ```stage=2```: 
  - counts frequencies of used codebook entries per speaker
  - if VERBOSE is true this stage also generates log-files
  - **if you need to extract features with a CPU, in the script ```local/codebook_freqs.py``` set ```device = torch.device('cpu')``` (default is ```device = torch.device('cuda')```)**
  - resulting files are stored in ```exp_*/logs/```, ```exp_*/numpy/``` and```exp_*/txt/```
- ```stage=3```: 
  - prepares and stores a similarty matrix in the folder ```exp_*/numpy/```
- ```stage=4```: 
  - performs a PCA on the similarity matrix and plots the PCA space
  - resulting ```*.png```-files are stored in ```exp_*/plots/analysis/```
- ```stage=5```: 
  - performs k-means on the resulting PCA space and assigns classes
  - **the parameter ```nclust``` in the script ```run.sh``` specifies the number of allowed clusters which should be modified depending on your task**
  - resulting ```*.png```-files (confusion matrices) are stored in ```exp_*/plots/kmenas/```

## Citation

If you use our code or data in your research, please cite our paper:

> ["Linke, J., Kadar, M., Dosinszky, G., Mihajlik, P., Kubin, G., Schuppler, B. (2023) What do self-supervised speech representations encode? An analysis of languages, varieties, speaking styles and speakers. Proc. INTERSPEECH 2023, 5371-5375, doi: 10.21437/Interspeech.2023-951"]

### BibTeX

```bibtex
@inproceedings{linke23_interspeech,
  author={Julian Linke and Mate Kadar and Gergely Dosinszky and Peter Mihajlik and Gernot Kubin and Barbara Schuppler},
  title={{What do self-supervised speech representations encode? An analysis of languages, varieties, speaking styles and speakers}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={5371--5375},
  doi={10.21437/Interspeech.2023-951}
}