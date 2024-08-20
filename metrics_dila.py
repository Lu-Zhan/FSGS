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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim

import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)

        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names


@torch.no_grad()
def evaluate(model_paths, scale, use_lpips=False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        # use_lpips = False

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ f"gt_{scale}"
            renders_dir = method_dir / f"test_preds_{scale}"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                render_image = renders[idx].to(device)
                gt_image = gts[idx].to(device)

                ssims.append(ssim(render_image, gt_image).cpu())
                psnrs.append(psnr(render_image, gt_image).cpu())

                # torch.cuda.empty_cache()

                if use_lpips:
                    try:
                        lpips_value = lpips_fn(renders[idx].to(device) * 2 - 1, gts[idx].to(device) * 2 - 1).data.cpu()
                    except:
                        lpips_value = -1.
                        # use_lpips = False
                else:
                    lpips_value = -1.

                # lpips_value = -1.
                lpipss.append(lpips_value)

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},})

            psnr_value = torch.tensor(psnrs).mean()
            ssim_value = torch.tensor(ssims).mean()
            lpips_value = torch.tensor(lpipss).mean()

            os.makedirs(scene_dir + f"/metrics", exist_ok=True)

            with open(scene_dir + f"/metrics/paper_r{scale}_{psnr_value:.4f}_{ssim_value:.4f}_{lpips_value:.4f}.txt", 'w+') as fp:
                fp.write(f"PSNR: {psnr_value}\n")
                fp.write(f"SSIM: {ssim_value}\n")
                fp.write(f"LPIPS: {lpips_value}\n")
                
        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    # torch.cuda.set_device(device)
    # lpips_fn = lpips.LPIPS(net='vgg').to(device)
    # lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=1)
    parser.add_argument('--lpips', type=int, default=0)
    
    args = parser.parse_args()

    if args.lpips:
        # lpips_fn = lpips.LPIPS(net='vgg').to(device)
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    print("Exp dir:", args.model_paths, "Resolution:", args.resolution)
    evaluate(args.model_paths, args.resolution, args.lpips)
