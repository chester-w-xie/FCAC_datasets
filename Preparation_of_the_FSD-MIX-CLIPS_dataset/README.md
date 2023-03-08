
 FSD-MIX-CLIPS(FMC) is an open dataset of programmatically mixed audio clips with a controlled level
 of polyphony and signal-to-noise ratio. More details about the FMC dataset, please refer [Wang et al.](https://ieeexplore.ieee.org/abstract/document/9632677)
 
Due to the large size of the FMC dataset, [Wang](https://y-wang.weebly.com/) did not upload it to the web, so you need to download the material provided by [Wang](https://y-wang.weebly.com/) to generate the data locally.

Specifically,
1. Download [FSD_MIX_SED.source](https://zenodo.org/record/5574135/files/FSD_MIX_SED.source.tar.gz?download=1) and [FSD_MIX_SED.annotations](https://zenodo.org/record/5574135/files/FSD_MIX_SED.annotations.tar.gz?download=1) to your local machine and unzip them.
You should get a structure of the directory as follows:
<pre>
dataset_root
├── FSD_MIX_SED.annotations
│    ├── base
│    |    ├── train (205,039 files)
│    |    |      ├── soundscape_0.jams
│    |    |      ├── .....
│    |    |      └── soundscape_205038.jams
│    |    ├── test (30,000 files)
│    |    |      ├── soundscape_0.jams
│    |    |      └── .....
│    |    └── val (30,000 files)
│    |          ├── soundscape_0.jams
│    |          └── .....
│    ├── test (8,000 files) # -novel-test
│    |    ├── soundscape_0.jams
│    |    ├── .....
│    |    └── soundscape_7999.jams
│    ├── val (8,000 files) # -novel-val
│    |    ├── soundscape_0.jams
│    |    ├── .....
│    |    └── soundscape_7999.jams
│    |
└── FSD_MIX_SED.source
    ├── foreground
    |    ├── train (59 folders)
    |    |      ├── 0
    |    |      ├── .....
    |    |      └── 58
    |    ├── test (59 folders)
    |    |      ├── 0
    |    |      └── .....
    |    └── val (59 folders)
    |          ├── 0
    |          └── .....
    ├── test (15 folders) # -novel-test
    |    ├── 74
    |    ├── .....
    |    └── 88
    ├── val (15 folders) # -novel-val
    |    ├── 59
    |    ├── .....
    |    └── 73
    └── background
           └── brownnoise (1 files)
                  └── brownnoise.wav

</pre>

2. FSD-MIX-SED is a dataset of soundscapes. Each soundscape contains n events from n different sound classes where n ranges from 1 to 5. 
Each sample of the FSD-MIX-CLIPS dataset is extracted from the soundscape in the FSD-MIX-SED dataset.
To generate the FSD-MIX-SED dataset, run the following command:
```
python generate_soundscapes.py \
--jamspath path to FSD_MIX_SED.annotations \
--sourcepath path to FSD_MIX_SED.source \
--savepath path to save the FSD-MIX-SED dataset
```

 3. Since there are duplicate annotations in the original annotation files of the FMC dataset, 
 we have removed the duplicate information. The revised annotation files are [here](./FSD_MIX_CLIPS.annotations_revised).

 4. With the revised annotation files and the FSD-MIX-SED dataset, we can generate the FMC dataset by running the following command:

```
python get_cilps_audio_or_openl3.py --annpath path to FSD_MIX_CLIPS.annotations_revised \
--audiopath path to FSD_MIX_SED.audio \
--savepath path to save the FSD-MIX-CLIPS dataset --data_type audio
```
The resulting FMC dataset will be saved in the corresponding path in the following directory structure:
<pre>
├── FSD_MIX_CLIPS_data # -  
    └── audio  # - audio samples
         ├── base
         |     ├── train
         |     |      ├── soundscape_205038_327222_1642.wav
         |     ├── val
         |     └── test
         ├── test
         └── full_filelist   # - path to read the sample and label, etc.
</pre>
By conducting a statistical analysis of the samples in the FMC dataset, we obtained the following statistics:

Number of samples in each subset:

|                          |  Base-train  |  Base-val  |   Base-test   |  Novel-val   | Novel-test  |
|:------------------------:|:------------:|:----------:|:-------------:|:------------:|:-----------:|
| sample with singel label |   351,781    |   51,889   |    50,550     |    13,358    |   12,605    |
| sample with multi-label  |    96,342    |   13,631   |    14,872     |    3,989     |    4,031    |
|          total           |   448,123    |   65,520   |    65,422     |    17,347    |   16,636    |
|    Ave num. per class    |    5,962     |    879     |      856      |     890      |     840     |
|[min, max] num. per class | [5774, 6160] | [810, 931] |  [801, 908]   |  [834, 937]  | [791, 871]  |

Note: Base-train, Base-val, and Base-test subsets contain sound samples of classes 0 to 58, 
and the samples in the three subsets do not overlap. The subset Novel-val contains sound samples of classes 59 to 73, 
and the subset Novel-test contains sound samples of classes 74 to 88. The specific names of the sounds can be found [here](https://github.com/chester-w-xie/FCAC_datasets/blob/main/vocabulary_of_FSC-89.json).

Considering the above statistics and the problem setting of FCAC, we have two options for constructing the FCAC dataset:

### Setup 1 (default setting):
- We select classes 0 to 58 as Base classes, classes 59 to 88 as Novel classes
- for each Base class, 800 samples for training (Sampling from Base-train)
  200 samples for validation (Sampling from Base-val), 200 samples for testing  (Sampling from Base-test)

- Merge the Novel-val and Novel-test subsets of the FMC into one dataset and use it as the Novel subset for FCAC. 
- for each Novel class, 500 samples for training and 200 samples for testing (Sampling from the Novel subset, respectively). The
sample in train and test subsets do not overlap.


### Setup 2 (more training and validation samples for base classes):
- We select classes 0 to 58 as Base classes, classes 59 to 88 as Novel classes
- for each Base class, 5000 samples for training (Sampling from Base-train)
  800 samples for validation (Sampling from Base-val), 200 samples for testing  (Sampling from Base-test)

- Merge the Novel-val and Novel-test subsets of the FMC into one dataset and use it as the Novel subset for FCAC. 
- for each Novel class, 500 samples for training and 200 samples for testing (Sampling from the Novel subset, respectively). The
sample in train and test subsets do not overlap.
