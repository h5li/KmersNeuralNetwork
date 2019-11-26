import pandas as pd
import numpy as np

train_data = pd.read_csv('../data/Kmers6_counts_600bp.csv')
train_reads = pd.read_csv('../data/Mouse_DMRs_counts_total.csv',header = None)
train_methys = pd.read_csv('../data/Mouse_DMRs_counts_methylated.csv',header = None)
train_methy_level = pd.read_csv('../data/Mouse_DMRs_methylation_level.csv',header = None)



