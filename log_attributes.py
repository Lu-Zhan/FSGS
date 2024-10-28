import os
import glob
import torch
import tqdm
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


def log_number(ckpt_dir, scenes, method):
    for scene in scenes:
        ckpt_paths = glob.glob(os.path.join(ckpt_dir, scene, "ckpt*.pth"))
        ckpt_paths.sort()

        nums = []
        iters = []

        gs_model = SimpleGaussianModel()

        for ckpt_path in tqdm.tqdm(ckpt_paths):
            ckpt, iter = torch.load(ckpt_path)
            gs_model.update(ckpt)

            nums.append(gs_model.get_xyz.shape[0])
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


if __name__ == "__main__":
    methods = ["3dgs", "fsgs_vanilla"]

    for method in methods:
        ckpt_dir = f"/home/titan/exps/exps_frgs/stat/few_shot/llff_view3/{method}"
        scenes = os.listdir(ckpt_dir)

        log_number(ckpt_dir, scenes, method)