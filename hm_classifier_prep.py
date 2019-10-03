import pandas as pd
import numpy as np

reads = pd.read_csv('classifier_all_reads.csv')

reads = reads.pivot_table(index='sample_id', columns='human_gene_symbol', values='CPM Reads', aggfunc=np.mean, fill_value=0.0)

reads.to_hdf('all_reads_matrix.h5', key='matrix')