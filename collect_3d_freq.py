import os
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib import colormaps

# /home/space/exps/fsgs_exps/llff_dila/prove/vanila_dila_0_den0.0005/trex/3D_frequency/10000.npz

def compute_freq(data, max_freq=3000, xs_step=1000):
    scales = data[:, :3]    # (n, 3)
    opacity = data[:, 3]    # (n,)

    scales = scales.mean(axis=1)    # (n,)

    mask = (opacity > 0.01) & (scales > 0)

    scales = scales[mask]
    opacity = opacity[mask]
    scales_freq = 1 / scales    # (n,)

    xs = np.linspace(0, max_freq, xs_step)
    ys = np.zeros(xs_step)

    for op, scale in zip(opacity, scales_freq):
        ys += op * stats.norm.pdf(xs, 0, scale)  # (1000,)
        
    return xs, ys


def compare_methods(method_paths, method_names, save_path, freq_split=[800, 1600]):
    colors = colormaps['viridis'].resampled(len(method_paths))
    colors = [colors(i / len(method_paths)) for i in range(len(method_paths))]

    data_freqs = []
    for data_path in method_paths:
        npy_list = glob.glob(os.path.join(data_path, '*.npy'))
        npy_list = sorted(npy_list, key=lambda x: float(x.split('/')[-1].split('.')[0]))
        freqs = []
        
        for npy_path in npy_list:
            data = np.load(npy_path)    # (n, 4)

            xs, ys = compute_freq(data)
            del data
            print(f'Finish {npy_path}')

            low_freq_index = np.where(xs < freq_split[0])
            mid_freq_index = np.where((xs >= freq_split[0]) & (xs < freq_split[1]))
            high_freq_index = np.where(xs >= freq_split[1])

            low_freq = ys[low_freq_index].sum()
            mid_freq = ys[mid_freq_index].sum()
            high_freq = ys[high_freq_index].sum()

            freqs.append([low_freq, mid_freq, high_freq])
        
        freqs = np.array(freqs) # (N, 3)
        iters = np.array([float(os.path.basename(x)[:-4]) for x in npy_list])   # (N, )

        data_freqs.append(freqs)

    data_freqs = np.array(data_freqs)    # (M, N, 3)

    # draw chart with multiple methods, each method uses different color in colormaps of `Jet'
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    for i, freqs in enumerate(data_freqs):
        ax[0].plot(iters, freqs[:, 0], label=f'{method_names[i]}', color=colors[i])
        ax[1].plot(iters, freqs[:, 1], label=f'{method_names[i]}', color=colors[i])
        ax[2].plot(iters, freqs[:, 2], label=f'{method_names[i]}', color=colors[i])
    
    ax[0].set_title('Low Frequency')
    ax[1].set_title('Mid Frequency')
    ax[2].set_title('High Frequency')

    fig.legend(method_names)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # save_path = os.path.join(save_dir, 'frequency_stat.png')
    plt.savefig(save_path)

    print(f'Save to {save_path}')


def compare_chart(method_paths, freq_split=[800, 1600]):
    colors = colormaps['viridis'].resampled(len(method_paths))
    colors = [colors(i / len(method_paths)) for i in range(len(method_paths))]

    data_freqs = []
    for data_path in method_paths:
        npy_list = glob.glob(os.path.join(data_path, '*.npy'))
        npy_list = sorted(npy_list, key=lambda x: float(x.split('/')[-1].split('.')[0]))
        freqs = []
        
        for npy_path in npy_list[-1:]:
            data = np.load(npy_path)    # (n, 4)

            xs, ys = compute_freq(data)
            print(f'Finish {npy_path}')

            low_freq_index = np.where(xs < freq_split[0])
            mid_freq_index = np.where((xs >= freq_split[0]) & (xs < freq_split[1]))
            high_freq_index = np.where(xs >= freq_split[1])

            low_freq = ys[low_freq_index].sum()
            mid_freq = ys[mid_freq_index].sum()
            high_freq = ys[high_freq_index].sum()

            freqs.append([low_freq, mid_freq, high_freq])
        
        print(data.shape[0])
        
        freqs = np.array(freqs) # (N, 3)

        np.savez(os.path.join(data_path, 'chart.npz'), data=freqs)

        # save mat file
        import scipy.io as sio
        sio.savemat(os.path.join(data_path, 'chart.mat'), {'data': freqs})


def draw_comparison(method_paths, method_names, save_path):
    colors = colormaps['jet'].resampled(len(method_paths))
    colors = [colors(i / len(method_paths)) for i in range(len(method_paths))]
    iters = np.linspace(0, 10000, 101)

    data_freqs = []

    for data_path in method_paths:
        data = np.load(os.path.join(data_path, 'chart.npz'))['data']
        data_freqs.append(data)

    data_freqs = np.array(data_freqs)    # (M, N, 3)

    # draw chart with multiple methods, each method uses different color in colormaps of `Jet'
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    for i, freqs in enumerate(data_freqs):
        ax[0].plot(iters, freqs[:, 0], label=f'{method_names[i]}', color=colors[i])
        ax[1].plot(iters, freqs[:, 1], label=f'{method_names[i]}', color=colors[i])
        ax[2].plot(iters, freqs[:, 2], label=f'{method_names[i]}', color=colors[i])
    
    ax[0].set_title('Low Frequency')
    ax[1].set_title('Mid Frequency')
    ax[2].set_title('High Frequency')

    fig.legend(method_names)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # save_path = os.path.join(save_dir, 'frequency_stat.png')
    plt.savefig(save_path)

    print(f'Save to {save_path}')


def draw_comparison(method_paths, method_names, save_path):
    # colors = colormaps['jet'].resampled(len(method_paths))
    # colors = [colors(i / len(method_paths)) for i in range(len(method_paths))]
    iters = np.linspace(0, 10000, 101)

    data_freqs = []

    for data_path in method_paths:
        data = np.load(os.path.join(data_path, 'chart.npz'))['data']
        data_freqs.append(data)

    data_freqs = np.array(data_freqs)    # (M, N, 3)

    # draw chart with multiple methods, each method uses different color in colormaps of `Jet'
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    for i, freqs in enumerate(data_freqs):
        ax[2].plot(iters, freqs[:, 2], label=f'{method_names[i]}', color=colors[i])
    
    ax[0].set_title('Low Frequency')
    ax[1].set_title('Mid Frequency')
    ax[2].set_title('High Frequency')

    fig.legend(method_names)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # save_path = os.path.join(save_dir, 'frequency_stat.png')
    plt.savefig(save_path)

    print(f'Save to {save_path}')
    
    

if __name__ == '__main__':
    data_dir = '/home/titan/exps_backup/fsgs_exps/llff/freq_stat'
    # data_dir = '/home/titan/exps_backup/corgs_exps'

    exp_names = ['vanilla_3dgs', '3dgs_0.3', 'fsgs_0', 'fsgs_0.2']
    # exp_names = ['llff_stat']
    data_names = ['flower']

    for data_name in data_names:
        data_paths = [os.path.join(data_dir, exp_name, data_name, 'frequency_data') for exp_name in exp_names]
        compare_chart(data_paths, freq_split=[800, 1600])