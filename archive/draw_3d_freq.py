import os
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib import colormaps

import scipy.io as sio

def draw_freq():
    iters = np.linspace(0, 10000, 101)

    data_freqs = []

    method_names = [f'freq_fewshot/{x}.mat' for x in ['3dgs', 'ours', 'fsgs', 'corgs']]

    for data_path in method_names:
        # load mat data
        data = sio.loadmat(data_path)['data']
        data_freqs.append(data)

    data_freqs = np.array(data_freqs)    # (M, N, 3)

    # draw chart with multiple methods, each method uses different color in colormaps of `Jet'
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    linewidth = 3

    for i, freqs in enumerate(data_freqs):
        if method_names[i] == 'freq_fewshot/corgs.mat':
            ax[0].plot(iters, freqs[:, 2] / 2.5, label=f'{i}', linewidth=linewidth)
        elif method_names[i] == 'freq_fewshot/3dgs.mat':
            ax[0].plot(iters, freqs[:, 2], label=f'{i}', linewidth=linewidth, linestyle='dashed')
        else:
            ax[0].plot(iters, freqs[:, 2], label=f'{i}', linewidth=linewidth)

    fig.legend(method_names)

    # /home/luzhan/Projects/gs/mip-splatting/freq.npy

    mip_data = np.load('/home/luzhan/Projects/gs/mip-splatting/freq.npy')

    xs = np.linspace(0, 3000, mip_data.shape[-1])
    for i, ys in enumerate(mip_data):
        if i == 0:
            ax[1].plot(xs, ys, label=f'{i}', linewidth=linewidth, linestyle='dashed')
        else:
            ax[1].plot(xs, ys, label=f'{i}',  linewidth=linewidth)

    plt.savefig('frequency_stat.svg', dpi=600, format='svg')
    plt.savefig('frequency_stat.png', dpi=600, format='png')
    
draw_freq()