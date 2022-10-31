# Details of the datasets for Few-shot class-incremental audio classification
ASVP@SCUT 👍👍👍🤙🤙🤙

This repository contains the datasets description of Nsynth-100 and Free-sound cilps of 89 classes (FSC-89) , which are used in the paper, 
"Few-shot class-incremental audio classification using adaptively-refined prototypes" (Submitted to ICASSP2023).

## Motivation for constructing the datasets

To study the Few-shot Class-incremental Audio Classification (FCAC) problem, we constructed the Nsynth-100 dataset and FSC-89 dataset
using the [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) data and the [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo) dataset as the source materials,
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


The method of constructing the datasets will be available when the final version of the paper is confirmed.

The NSynth dataset is an audio dataset containing 305,979 musical notes, each with a unique pitch, timbre, and envelope. 
Those musical notes are belonging to 1,006 musical instruments.






## Contact
Wei Xie

South China University of Technology, Guangzhou, China
 
chester.w.xie@gmail.com


## Citation
Please cite our paper if you find the datasets are useful for your research.

Wei Xie, Yanxiong Li, Qianhua He. "Few-shot class-incremental audio classification using adaptively-refined prototypes", xxx, 2023


