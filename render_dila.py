#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

torch.set_float32_matmul_precision("medium")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # check if the render result exists
        # if os.path.exists(os.path.join(render_path, '{0:05d}'.format(idx) + ".png")):
        #     continue

        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torch.cuda.empty_cache()


@torch.no_grad()
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    gaussians = GaussianModel(dataset)

    gaussians.disable_grad()
    dataset.load_allres = False

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, used_sets=["test"])
    scale_factor = dataset.resolution
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    if not skip_train:
        views = scene.getTrainCameras()
        print('Renderer size:', views[0].image_width)
        render_set(dataset.model_path, "train", scene.loaded_iter, views, gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

    if not skip_test:
        views = scene.getTestCameras()
        print('Renderer size:', views[0].image_width)
        render_set(dataset.model_path, "test", scene.loaded_iter, views, gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--redo", action="store_true")
    args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # if not args.redo:
    #     # check if the render result exists, outputs/benchmark_360v2_stmt/dila_0.1/bicycle/test/ours_30000/gt_1
    #     save_path = os.path.join(args.model_path, f"test/ours_30000/gt_{args.resolution}/00000.png")

    #     if os.path.exists(save_path):
    #         print(f'Results already exist, {save_path}')
    #         exit(0)
    
    print("Rendering:", args.model_path, "Resolution:", args.resolution)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)