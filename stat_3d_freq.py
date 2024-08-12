import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps



# /home/space/exps/fsgs_exps/llff_dila/prove/vanila_dila_0_den0.0005/trex/3D_frequency/10000.npz

def draw_chart(data_path, save_dir, freq_split=[800, 1600]):
    npy_list = glob.glob(os.path.join(data_path, '*.npy'))

    npy_list = sorted(npy_list, key=lambda x: float(x.split('/')[-1].split('.')[0]))

    freqs = []
    
    for npy_path in npy_list:
        data = np.load(npy_path)    # (3, 1000, 2)
        data = data.mean(axis=0)    # (1000, 2)
        xs = data[:, 0]
        ys = data[:, 1]

        low_freq_index = np.where(xs < freq_split[0])
        mid_freq_index = np.where((xs >= freq_split[0]) & (xs < freq_split[1]))
        high_freq_index = np.where(xs >= freq_split[1])

        low_freq = ys[low_freq_index].sum()
        mid_freq = ys[mid_freq_index].sum()
        high_freq = ys[high_freq_index].sum()

        freqs.append([low_freq, mid_freq, high_freq])
    
    # draw chart
    freqs = np.array(freqs) # (N, 3)
    iters = np.array([float(os.path.basename(x)[:-4]) for x in npy_list])   # (N, )

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].plot(iters, freqs[:, 0], label='low freq')
    ax[0].set_title('Low Frequency')

    ax[1].plot(iters, freqs[:, 1], label='mid freq')
    ax[1].set_title('Mid Frequency')

    ax[2].plot(iters, freqs[:, 2], label='high freq')
    ax[2].set_title('High Frequency')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'frequency_stat.png')
    plt.savefig(save_path)

    print(f'Save to {save_path}')
    

if __name__ == '__main__':
    data_dir = '/home/space/exps/fsgs_exps/llff_dila/freq_stat'
    exp_name = 'vanilla_dila0_den0.0005'
    data_name = 'trex'

    save_dir = os.path.join(data_dir, exp_name, data_name)
    data_path = os.path.join(data_dir, exp_name, data_name, 'frequency_data')

    draw_chart(data_path, save_dir)