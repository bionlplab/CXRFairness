# CXRFairness

# Improving Fairness of Automated Chest X-ray Diagnosis by Contrastive Learning

## Datasets

The first dataset is provided by Medical Imaging and Data Resource Center (MIDRC) and is available through this website (https://data.midrc.org/). The NIH-CXR dataset is available in this website (https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest),

## Getting started

### Prerequisites

* python >=3.6
* pytorch = 1.11.0
* torchvision = 0.12.0
* sklearn
* pandas
* opencv
* skimage
* tqdm
* json
* pickle

### Quickstart

I used the experiment on the MIMIC-CXR dataset on the intersectional groups as an example.

```sh
python train_mimic_sex.py
then
python train_finetune_race.py
```

### Reference



### Acknowledgment

This work was supported by the National Library of Medicine under Award No. 4R00LM013001, NSF CAREER Award No. 2145640, and Amazon Research Award.
