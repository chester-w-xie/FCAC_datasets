### Statistics of the NSynth dataset
|                                             |   train   |    val    |   test    |
|:-------------------------------------------:|:---------:|:---------:|:---------:|
|        Num. of classes (Instruments)        |    953    |    53     |    53     | 
|            Total num of samples             |  289,205  |  12,678   |   4096    | 
|        Ave num. of samples per class        |    303    |    239    |    125    | 
|    [min, max] num. of samples per class     | [30, 440] | [83, 348] | [22, 125] |

Note: The classes of the validation set and the test set are identical, 
while the classes of the training set do not overlap with the validation/test set.

Statistics on the number of samples of the combination of the validation and test sets:
- Number of classes with more than 100 samples: 53
- Number of classes with more than 150 samples: 52
- Number of classes with more than 200 samples: 45
- Number of classes with more than 250 samples: 32
- Number of classes with more than 300 samples: 31

Statistics on the samples of the training set:
- Number of classes with more than 450 samples: 0
- Number of classes with more than 440 samples: 377
- Number of classes with more than 400 samples: 382

➢ Based on the above statistics, we have the following options for constructing the dataset：

#### setup 1 (NSynth-100, default setting, used in our ICASSP2023 paper):
Merging the test set and the validation set to obtain a dataset of novel audio classes,
from which 45 classes with more than 200 samples are randomly selected as the novel classes. 
Then 100 and 100 samples per novel class are randomly selected as the training and testing samples for that class, respectively. 

We randomly selected 55 classes with more than 400 samples from the training set as the base classes. 
Then 200,100, and 100 samples per base class are randomly selected as the training, validation, and testing sets, respectively. 

#### setup 2 (NSynth-200):
The configurations are the same as setup 1, except that the number of base classes is 155.

#### setup 3 (NSynth-300):
The configurations are the same as setup 1, except that the number of base classes is 255.

#### setup 4 (NSynth-400):
The configurations are the same as setup 1, except that the number of base classes is 355.
