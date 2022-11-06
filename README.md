# Details of the datasets for Few-shot class-incremental audio classification
ASVP@SCUT 👍👍👍🤙🤙🤙

This repository contains the datasets description of Nsynth-100 and Free-sound cilps of 89 classes (FSC-89) , which are used in the paper, 
"Few-shot class-incremental audio classification using adaptively-refined prototypes" (Submitted to ICASSP2023).

Motivation for constructing the datasets: 

To study the Few-shot Class-incremental Audio Classification (FCAC) problem, we constructed the Nsynth-100 dataset and FSC-89 dataset
using the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset and the [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo) dataset as the source materials,
respectively.



**Table of Contents**
- [Statistics on the datasets](#statistics-on-the-datasets)
- [Preparation of the datasets](#preparation-of-the-datasets)
- [Contact](#contact)
- [Citation](#citation)


## Statistics on the datasets
|                                                                 |                  Nsynth-100                   |                    FSC-89                    |
|:---------------------------------------------------------------:|:---------------------------------------------:|:--------------------------------------------:|
|                          Type of audio                          |              Musical instruments              |                  Free sound                  |
|                         Num. of classes                         | 100 (55 of base classes, 45 of novel classes) | 89 (59 of base classes, 30 of novel classes) |
| Num. of training / validation / testing samples per base class  |                200 / 100 / 100                |               800 / 200 / 200                |
| Num. of training / validation / testing samples per novel class |               100 / none / 100                |               500 / none / 200               |
|                     Duration of the sample                      |               All in 4 seconds                |               All in 1 second                |
|                       Sampling frequency                        |                 All in 16K Hz                 |               All in 44.1K Hz                |

## Preparation of the datasets


The NSynth dataset is an audio dataset containing 305,979 musical notes, each with a unique pitch, timbre, and envelope. 
Those musical notes are belonging to 1,006 musical instruments. 

Before constructing the Nsynth-100 dataset, we first conduct some statistical analysis on the Nsynth dataset, see [here](/Statistics_of_the_Nsynth_dataset.md).

Based on the statistical results, we obtain the Nsynth-100 dataset by the following steps:

1. Download [Train set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz), [Valid set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz), and [test set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz) of the Nsynth dataset to your local machine and unzip them.
You should get a structure of the directory as follows:
<pre>
Your dataset root 
├── nsynth-train　 # Training set of the Nsynth dataset
│    ├── audio
│    |    ├── bass_acoustic_000-024-025.wav
│    |    └── ....
│    └── examples.json  # meta file of the training set
│
├── nsynth-val  # Validation set of the Nsynth dataset
│    ├── audio
│    |    ├── bass_electronic_018-022-025.wav
│    |    └── ....
│    └── examples.json
│
└── nsynth-test # Test set of the Nsynth dataset
     ├── audio
     |    ├── bass_electronic_018-022-100.wav
     |    └── ....
     └── examples.json
</pre>
2. Download the meta files of the nsynth-100/200/300/400 datasets and save them with the audio datasets according to the following directory structure:
<pre>
Your dataset root
├── nsynth-100-meta
│    ├── nsynth-100_train.csv # containing information of all training samples from the base and novel classes
│    ├── nsynth-100_val.csv  #　containing information of all validation samples from the base classes
│    ├── nsynth-100_test.csv　# containing information of all test samples from the old and novel classes
│    └── nsynth-100_vocab.json 　# label vocabulary of the dataset
│    
├── nsynth-200-meta
│    ├── nsynth-200_train.csv #  
│    ├── nsynth-200_val.csv
│    ├── nsynth-200_test.csv
│    └── nsynth-200_vocab.json
│    
├── nsynth-300-meta
│    ├── nsynth-300_train.csv #  
│    ├── nsynth-300_val.csv
│    ├── nsynth-300_test.csv
│    └── nsynth-300_vocab.json
│       
└── nsynth-400-meta
     ├── nsynth-400_train.csv #  
     ├── nsynth-400_val.csv
     ├── nsynth-400_test.csv
     └── nsynth-400_vocab.json

</pre>






## Contact
Wei Xie

South China University of Technology, Guangzhou, China
 
chester.w.xie@gmail.com


## Citation
Please cite our paper if you find the datasets are useful for your research.

Wei Xie, Yanxiong Li, Qianhua He. "Few-shot class-incremental audio classification using adaptively-refined prototypes", xxx, 2023


