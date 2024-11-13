import os
import glob
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt


def get_image(data_dir, mode, resolutions):
    images = OrderedDict()

    for r in resolutions:
        image_paths = glob.glob(os.path.join(data_dir, f"{mode}_{r}", "*.png"))
        image_paths = sorted(image_paths)
        images[r] = [Image.open(x) for x in image_paths]
    
    return images


def compare_results(collected_images):
    compared_images = OrderedDict()

    for key in collected_images.keys():
        compared_images[key] = compare_image(collected_images['gt'], collected_images[key])
    
    return compared_images


def compare_image(gts, preds):
    comp = {}
    for r in gts.keys():
        comp[r] = []

        for gt, pred in zip(gts[r], preds[r]):
            
            if gt.size != pred.size:
                gt = gt.resize(pred.size)

            psnr = get_psnr(np.array(gt) / 255, np.array(pred) / 255)
            err_map = np.abs(np.array(gt) / 255 - np.array(pred) / 255)
            comp[r].append((err_map, psnr))

    return comp


def get_psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)

    if mse == 0:
        return 0
    
    return -10 * np.log10(mse)


if __name__ == "__main__":
    base_dir = "/home/titan/exps/exps_frgs/byproduct"
    exp_names = ["mip360_view12"]
    method_names = ["3dgs_vanilla", "fsgs_vanilla", "mipsplatting", "lsy_den500_0.1"]
    # resolutions = [1, 2, 4, 8]
    resolutions = [8]

    # scene_names = os.listdir(os.path.join(base_dir, exp_names[0], method_names[0]))
    # scene_names.sort()

    scene_names = ["room", "counter", "bonsai", "bicycle"]

    for exp_name in exp_names:
        for scene_name in scene_names:
            collected_images = OrderedDict()

            for method_name in method_names:
                print(f"Processing {exp_name} {scene_name} {method_name}")
                data_dir = os.path.join(base_dir, exp_name, method_name, scene_name, "test", "ours_30000")

                if method_name == method_names[0]:
                    gt_images = get_image(data_dir, 'gt', resolutions)
                    collected_images['gt'] = gt_images
                
                pred_images = get_image(data_dir, 'test_preds', resolutions)
                collected_images[method_name] = pred_images
            
            compared_images = compare_results(collected_images)

            # plot images
            for r in resolutions:
                num_image = len(collected_images['gt'][r])
                num_col = len(collected_images.keys())

                all_rows = []
                rows = []
                for i in range(num_col):
                    image_row = []
                    comp_row = []
                    psnrs = []
                    for j, method_name in enumerate(collected_images.keys()):
                        image_row.append(np.array(collected_images[method_name][r][i]) / 255.)
                        comp_row.append(compared_images[method_name][r][i][0])
                        psnrs.append(compared_images[method_name][r][i][1])
                    
                    h, w, c = image_row[0].shape

                    image_row = np.concatenate(image_row, axis=1)
                    comp_row = np.concatenate(comp_row, axis=1)

                    blank_row = np.zeros((10, w, c))

                    for idx, psnr in enumerate(psnrs):
                        cv2.putText(blank_row, f"PSNR: {psnr:.4f}", (0, w * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    row = np.concatenate([image_row, blank_row, comp_row], axis=0)
                    rows.append(row)


                plt.tight_layout()
                save_path = f"stat/vis_byproduct/{exp_name}/{scene_name}_r{r}_{'_'.join(method_names)}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)

                print(f'finish plotting {scene_name}-r{r}')