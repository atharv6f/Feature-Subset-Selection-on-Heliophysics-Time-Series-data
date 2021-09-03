import os

import numpy as np

partition_1_path = os.path.join(os.getcwd(), "mvts_fss_scs\\data", "partition1")
partition_2_path = os.path.join(os.getcwd(), "mvts_fss_scs\\data", "partition2")

flare_path = os.path.join(partition_1_path,"FL")
non_flare_path = os.path.join(partition_1_path,"NF")


train_path = partition_1_path
test_path = partition_2_path


DATA_PATH = "home/ayeolekar1/mvts_fss_scs/mvts_fss_scs/preprocessed_data_samples"
PREPROCESSED_DATA_SAMPLES = "/home/ayeolekar1/mvts_fss_scs/mvts_fss_scs/preprocessed_data_samples"
SAMPLED_DATA_SAMPLES = "home/ayeolekar1/mvts_fss_scs/mvts_fss_scs/sampled_data_samples"
RESULTS = "/home/ayeolekar1/mvts_fss_scs/Results"



part1 = np.load(os.path.join(DATA_PATH,"partition1_sample.npz"))



# DATA_PATH = /home/ayeolekar1/mvts_fss_scs/mvts_fss_scs/preprocessed_data_samples
# PREPROCESSED_DATA_SAMPLES = /"home/ayeolekar1/mvts_fss_scs/mvts_fss_scs/preprocessed_data_samples"
# SAMPLED_DATA_SAMPLES = "/home/ayeolekar1/mvts_fss_scs/mvts_fss_scs/sampled_data_samples"
# RESULTS = /home/ayeolekar1/mvts_fss_scs/Results
