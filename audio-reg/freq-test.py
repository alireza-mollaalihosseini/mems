import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import soundfile as sf


train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
train_filenames = np.loadtxt(train_files, dtype=str)

data, sample_rate = sf.read(train_filenames[0])

print(sample_rate)