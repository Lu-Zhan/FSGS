import os
import glob
import torch
import tqdm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import strip_symmetric, build_scaling_rotation, chamfer_dist


class SimpleGaussianModel:
    def __init__(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def update(self, model_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale
        ) = model_args
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # w = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def obtain_number(self, ckpt_path):
        ckpt, iter = torch.load(ckpt_path)
        self.update(ckpt)

        return self.get_xyz.shape[0], iter

    def obtain_scale(self, ckpt_path):
        ckpt, iter = torch.load(ckpt_path)
        self.update(ckpt)

        scale = self.get_scaling.mean(dim=-1).data.cpu().numpy()
        return scale, iter
    
    def obtain_opacity(self, ckpt_path):
        ckpt, iter = torch.load(ckpt_path)
        self.update(ckpt)

        opacity = self.get_opacity.data.cpu().numpy()[:, 0]
        return opacity, iter
    
    def obtain_freq(self, ckpt_path, max_freq=3000, xs_step=1000):
        ckpt, iter = torch.load(ckpt_path)
        self.update(ckpt)

        scales = self.get_scaling.mean(dim=-1).data.cpu().numpy()
        opacity = self.get_opacity.data.cpu().numpy()[:, 0]

        mask = (opacity > 0.01) & (scales > 0)

        scales = scales[mask]
        opacity = opacity[mask]
        scales_freq = 1 / scales    # (n,)

        xs = np.linspace(0, max_freq, xs_step)
        ys = np.zeros(xs_step)

        for op, scale in zip(opacity, scales_freq):
            ys += op * stats.norm.pdf(xs, 0, scale)  # (1000,)
            
        return ys, xs, iter


def clip_value(data):
    std = data.std()
    mean = data.mean()

    return np.clip(data, 0, mean + 3 * std)


def logging_2d(ckpt_dir, scenes, method):
    for scene in scenes:
        ckpt_paths = glob.glob(os.path.join(ckpt_dir, scene, "ckpt*.pth"))
        ckpt_paths.sort()

        nums = []
        iters = []

        gs_model = SimpleGaussianModel()

        for ckpt_path in tqdm.tqdm(ckpt_paths):
            num, iter = gs_model.obtain_number(ckpt_path)

            nums.append(num)
            iters.append(iter)
        
        # plot number/iter
        plt.close()
        plt.plot(iters, nums, label=scene)
        plt.xlabel("iter")
        plt.ylabel("num")
        plt.title(f"{method}-{scene}")
        plt.legend()

        save_path = f"stat/num_iter/{scene}_{method}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    
def logging_scale(ckpt_dir, scenes, method):
    for scene in scenes:
        ckpt_paths = glob.glob(os.path.join(ckpt_dir, scene, "ckpt*.pth"))
        ckpt_paths.sort()
        ckpt_paths = ckpt_paths[-1:]

        gs_model = SimpleGaussianModel()

        for ckpt_path in tqdm.tqdm(ckpt_paths):
            scale, iter = gs_model.obtain_scale(ckpt_path)

            # plot number/iter
            plt.close()
            plt.hist(scale, label=scene, bins=1000)
            plt.ylabel("scale")
            plt.title(f"{method}-{scene}")
            plt.legend()

            save_path = f"stat/scale_hist_normalized/{scene}/{iter}_{method}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)


def logging_scale_with_op(ckpt_dir, scenes, method):
    for scene in scenes:
        ckpt_paths = glob.glob(os.path.join(ckpt_dir, scene, "ckpt*.pth"))
        ckpt_paths.sort()
        ckpt_paths = ckpt_paths[-1:]

        gs_model = SimpleGaussianModel()

        for ckpt_path in tqdm.tqdm(ckpt_paths):
            scale, iter = gs_model.obtain_scale(ckpt_path)
            op, iter = gs_model.obtain_opacity(ckpt_path)

            scale_with_op = scale * op
            scale_with_op = clip_value(scale_with_op)

            # plot number/iter
            plt.close()
            plt.hist(scale_with_op, label=scene, bins=1000)
            plt.ylabel("scale_with_op")
            plt.title(f"{method}-{scene}")
            plt.legend()

            save_path = f"stat/scale_op_hist_normalized/{scene}/{iter}_{method}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)


def logging_inv_scale_with_op(ckpt_dir, scenes, method):
    for scene in scenes:
        ckpt_paths = glob.glob(os.path.join(ckpt_dir, scene, "ckpt*.pth"))
        ckpt_paths.sort()
        ckpt_paths = ckpt_paths[-1:]

        gs_model = SimpleGaussianModel()

        for ckpt_path in tqdm.tqdm(ckpt_paths):
            scale, iter = gs_model.obtain_scale(ckpt_path)
            op, iter = gs_model.obtain_opacity(ckpt_path)

            inv_scale_with_op = 1 / scale * op
            inv_scale_with_op = clip_value(inv_scale_with_op)

            # plot number/iter
            plt.close()
            plt.hist(inv_scale_with_op, label=scene, bins=1000)
            plt.ylabel("inv_scale_with_op")
            plt.title(f"{method}-{scene}")
            plt.legend()

            save_path = f"stat/inv_scale_op_hist_normalized/{scene}/{iter}_{method}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)


def logging_inv_scale_with_large_op(ckpt_dir, scenes, method, th=0.8):
    for scene in scenes:
        ckpt_paths = glob.glob(os.path.join(ckpt_dir, scene, "ckpt*.pth"))
        ckpt_paths.sort()
        ckpt_paths = ckpt_paths[-1:]

        gs_model = SimpleGaussianModel()

        for ckpt_path in tqdm.tqdm(ckpt_paths):
            scale, iter = gs_model.obtain_scale(ckpt_path)
            op, iter = gs_model.obtain_opacity(ckpt_path)

            mask_op = op > th

            inv_scale_with_op = 1 / scale
            inv_scale_with_op = inv_scale_with_op[mask_op]
            inv_scale_with_op = clip_value(inv_scale_with_op)

            # plot number/iter
            plt.close()
            plt.hist(inv_scale_with_op, label=scene, bins=1000)
            plt.ylabel("inv_scale_with_op")
            plt.title(f"{method}-{scene}")
            plt.legend()

            save_path = f"stat/inv_scale_large_op_hist_normalized/{scene}/{iter}_{method}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)


def logging_freq_1d(ckpt_dir, scenes, method):
    for scene in scenes:
        ckpt_paths = glob.glob(os.path.join(ckpt_dir, scene, "ckpt*.pth"))
        ckpt_paths.sort()
        ckpt_paths = ckpt_paths[-1:]

        gs_model = SimpleGaussianModel()

        for ckpt_path in tqdm.tqdm(ckpt_paths, desc=f"{method}_{scene}"):
            ys, xs, iter = gs_model.obtain_freq(ckpt_path)

            ys = np.log10(2 * ys + 1)
            # plot number/iter
            plt.close()
            # plt.hist(inv_scale_with_op, label=scene, bins=1000)
            plt.plot(xs, ys, label=scene)
            plt.ylabel("amplitude")
            plt.xlabel("frequency")
            plt.title(f"{method}-{scene}")
            plt.legend()

            save_path = f"stat/freq/{scene}/{iter}_{method}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)


def compare_freq_1d(ckpt_dir, scenes, methods):
    for scene in scenes:
        print('processing', scene)
        gs_model = SimpleGaussianModel()

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        for method in methods:
            ckpt_paths = glob.glob(os.path.join(ckpt_dir, method, scene, "ckpt*.pth"))
            ckpt_paths.sort()
            ckpt_path = ckpt_paths[-1]
            ys, xs, iter = gs_model.obtain_freq(ckpt_path)

            ys = np.log10(10 * ys + 1)

            num_split = len(xs) // 3
            freq_names = ["low", "mid", "high"]
            for idx in range(3):
                start_freq = idx * num_split
                end_freq = (idx + 1) * num_split
                axs[idx].plot(xs[start_freq: end_freq], ys[start_freq: end_freq], label=method)

                axs[idx].set_ylabel("amplitude")
                axs[idx].set_xlabel("frequency")
                axs[idx].set_title(freq_names[idx])

        plt.legend()

        save_path = f"stat/comparison_freq/final/{scene}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

        plt.close()


if __name__ == "__main__":
    methods = ["3dgs", "fsgs_vanilla"]

    # for method in methods:
    #     ckpt_dir = f"/home/titan/exps/exps_frgs/stat/few_shot/llff_view3/{method}"
    #     scenes = os.listdir(ckpt_dir)

    #     logging_freq_1d(ckpt_dir, scenes, method)

    ckpt_dir = "/home/titan/exps/exps_frgs/stat/few_shot/llff_view3"
    scenes = os.listdir(os.path.join(ckpt_dir, "3dgs"))
    compare_freq_1d(ckpt_dir, scenes, methods)
