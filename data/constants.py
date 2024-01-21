import os

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/ 
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = "C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/downloads/CUB/"
CUB_PROCESSED_DIR = "C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/downloads/CUB/class_attr_data_10"


# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/downloads/release_v0/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/downloads/archive/"


# BRODEN concept bank
BRODEN_CONCEPTS = "C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/downloads/broden_concepts/"

# ESC 50 Environmental sounds
ESC_50 = "C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/downloads/ESC-50-master"
ESC_50_META = os.path.join(ESC_50, "meta")