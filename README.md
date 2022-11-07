# Details of the datasets for Few-shot class-incremental audio classification (Nsynth-100 & FSC-89)
ASVP@SCUT 👍👍👍🤙🤙🤙

This repository contains the description of Nsynth-100 and Free-sound cilps of 89 classes (FSC-89) , which are proposed in the paper, 
"Few-shot class-incremental audio classification using adaptively-refined prototypes" (Submitted to ICASSP2023).

Motivation for constructing the datasets: 

To study the Few-shot Class-incremental Audio Classification (FCAC) problem, we constructed the Nsynth-100 dataset and FSC-89 dataset
using partial samples from the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset and the [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo) dataset as the source materials,
respectively.



**Table of Contents**
- [Statistics on the datasets](#statistics-on-the-datasets)
- [Preparation of the Nsynth-100 dataset](#preparation-of-the-Nsynth-100-dataset)
- [Preparation of the FSC-89 dataset](#preparation-of-the-fsc-89-dataset)
- [Acknowledgment](#acknowledgment)
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

## Preparation of the Nsynth-100 dataset


The NSynth dataset is an audio dataset containing 306,043 musical notes, each with a unique pitch, timbre, and envelope. 
Those musical notes are belonging to 1,006 musical instruments. 

Before constructing the Nsynth-100 dataset, we first conduct some statistical analysis on the Nsynth dataset, see [here](/Statistics_of_the_Nsynth_dataset.md).

Based on the statistical results, we obtain the Nsynth-100 dataset by the following steps:

1. Download [Train set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz), [Valid set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz), and [test set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz) of the Nsynth dataset to your local machine and unzip them.
You should get a structure of the directory as follows:
<pre>
Your dataset root (Nsynth_audio_for_FCAC)
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
2. Download the meta files for FCAC from [here](./Nsynth_meta_for_FCAC) to your local machine and unzip them.
You should get a structure of the directory as follows:
<pre>
Your dataset root (Nsynth_meta_for_FCAC)
├── nsynth-100-fs-meta
│    ├── nsynth-100-fs_train.csv # containing information of all training samples from the base and novel classes
│    ├── nsynth-100-fs_val.csv  #　containing information of all validation samples from the base classes
│    ├── nsynth-100-fs_test.csv　# containing information of all test samples from the old and novel classes
│    └── nsynth-100-fs_vocab.json 　# label vocabulary of the dataset
│    
├── nsynth-200-fs-meta
│    ├── nsynth-200-fs_train.csv #  
│    ├── nsynth-200-fs_val.csv
│    ├── nsynth-200-fs_test.csv
│    └── nsynth-200-fs_vocab.json
│    
├── nsynth-300-fs-meta
│    ├── nsynth-300-fs_train.csv #  
│    ├── nsynth-300-fs_val.csv
│    ├── nsynth-300-fs_test.csv
│    └── nsynth-300-fs_vocab.json
│       
└── nsynth-400-fs-meta
     ├── nsynth-400-fs_train.csv #  
     ├── nsynth-400-fs_val.csv
     ├── nsynth-400-fs_test.csv
     └── nsynth-400-fs_vocab.json

</pre>

3. Run the following script to load the Nsynth-100 dataset:
```
python Load_nsynth_data_for_FCAC.py --metapath path to Nsynth_audio_for_FCAC folder --audiopath path to Nsynth_meta_for_FCAC folder --num_class 100 --base_class 55

```

## Preparation of the FSC-89 dataset

1. Since the FSC-89 dataset is extracted from the FSD-MIX-CLIPS dataset, we need to prepare the FSD-MIX-CLIPS dataset first. See the instructions in
[here](./Preparation_of_the_FSD-MIX-CLIPS_dataset/README.md).

2. Download the meta file of FSC-89 dataset from [here](./FSC-89-meta), You should get a structure of the directory as follows:

<pre>
FSC-89-meta   
   ├── setup1  # 
   |     ├── Fsc89-setup1-fsci_train.csv # -  
   |     ├── Fsc89-setup1-fsci_val.csv  # -  
   |     └── Fsc89-setup1-fsci_test.csv # -  
   |
   └── setup2 # -  
         ├── Fsc89-setup2-fsci_train.csv # -  
         ├── Fsc89-setup2-fsci_val.csv  # -  
         └── Fsc89-setup2-fsci_test.csv # -  

</pre>

3. Run the following script to load the FSC-89 dataset:

```
python load_fsc_89_data_for_FCAC.py --metapath path to FSC-89-meta folder \
--datapath path to FSD-MIX-CLIPS_data folder --data_type audio --setup setup1
```

## Acknowledgment
Our project references the codes in the following repos.

- [rethink-audio-fsl](https://github.com/wangyu/rethink-audio-fsl)
- [SPPR](https://github.com/zhukaii/SPPR)

## Contact
Wei Xie

South China University of Technology, Guangzhou, China
 
chester.w.xie@gmail.com


## Citation
Please cite our paper if you find the datasets are useful for your research.

Wei Xie, Yanxiong Li, Qianhua He. "Few-shot class-incremental audio classification using adaptively-refined prototypes", xxx, 2023


