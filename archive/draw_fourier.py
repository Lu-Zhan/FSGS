import numpy as np
import matplotlib.pyplot as plt



def get_gaussian_and_freq(std=1):
    # Create a time vector from 0 to 10 with 1000 points
    x = np.linspace(0, 7, 1000)
    
    # creat a gaussian function with std=std and mu=0
    ty = np.exp(-0.5 * x**2 / std**2)

    fy = (2 * np.pi* std**2)**-0.5 * np.exp(-0.5 * x**2 * std**2)

    return x, ty, fy


def draw(stds=[0.5]):
    colors = ['r', 'g', 'b', 'm', 'y', 'c']

    for idx, std in enumerate(stds):
        get_gaussian_and_freq(std=std)
    
        x, ty, fy = get_gaussian_and_freq(std=std)

        plt.plot(x, ty, label=f'time-scale={std}', linestyle='--', c=colors[idx])
        plt.plot(x, fy, label=f'freq-scale={std}', c=colors[idx])

    plt.legend()

    plt.savefig('fourier.svg', dpi=600)

# draw(stds=[0.5, 0.75, 1, 1.25, 1.5, 1.75])

draw(stds=[0.5, 1, 1.5, 2])