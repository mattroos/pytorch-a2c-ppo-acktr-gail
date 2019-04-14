# plot.py

import numpy as np
import pandas as pd
import os
import sys
import time
import pdb
import matplotlib.pyplot as plt
plt.ion()

if len(sys.argv) > 1:
    k_group = int(sys.argv[1])
else:
    k_group = 100

if len(sys.argv) > 2:
    metric = sys.argv[2]
else:
    metric = 'r'

if len(sys.argv) > 3:
    dir_files = sys.argv[3]
else:
    dir_files = '/tmp/gym/'



def group(x, k=100):
    x = x.astype(np.float)
    n = len(x)
    m = n//k
    y = x[:m*k]
    y[np.argwhere(np.isinf(y))] = np.nan    # ignore inf and nan
    # y[np.argwhere(y==0)] = np.nan    # ignore inf and nan
    y = np.reshape(y, (k,m), order='F')
    #y = np.mean(y, axis=0)
    y = np.nanmean(y, axis=0)
    #y = y - np.nanmean(y)
    #y = y / np.nanstd(y)
    return y


list_filenames = []
for file in os.listdir(dir_files):
    if file.endswith('.csv'):
        list_filenames.append(os.path.join(dir_files, file))

plt.figure(1)
len_max = 0
data = []
for fn in list_filenames:
    df = pd.read_csv(fn, comment='#')
    data.append(group(df[metric].values, k_group))
    len_data = len(data[-1])
    if len_data > len_max:
        len_max = len_data
    x = np.arange(len_data) * k_group
    plt.plot(x, data[-1], 'o', markerfacecolor='none')
    # noise = np.random.rand(*data[-1].shape)*5
    # plt.plot(x, data[-1]+noise, 'o', markerfacecolor='none')

if k_group > 1:
    data_avg = np.full((len_max, len(data)), np.nan)
    for i, d in enumerate(data):
        data_avg[:len(d), i] = d
    data_avg = np.nanmean(data_avg, axis=1)
    x = np.arange(len_max) * k_group
    plt.plot(x, data_avg, '.-')

# plt.ylim(0, 200)
plt.grid(True)
plt.xlabel('Evaluation Number')
plt.ylabel('Total Reward')
timestamp = os.path.getmtime(list_filenames[0])
date_time = time.strftime("%x %X", time.localtime(timestamp))
plt.title('SaccadeDigit-v0 Learning, %s' % (date_time))
