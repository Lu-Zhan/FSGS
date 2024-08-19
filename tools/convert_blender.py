import numpy as np
import json
import os
import imageio
import math

H = 800
W = 800

base_dir = '/home/luzhan/nerf_synthetic'

blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
for scene in scenes:
    # os.chdir(os.path.join(base_dir, scene))
    data_dir = os.path.join(base_dir, scene)

    fnames = list(sorted(os.listdir(os.path.join(data_dir, "train"))))
    fname2pose = {}

    with open(os.path.join(data_dir, "transforms_train.json"), "r") as f:
        meta = json.load(f)

    fx = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])  # original focal length
    if "camera_angle_y" in meta:
        fy = 0.5 * H / np.tan(0.5 * meta["camera_angle_y"])  # original focal length
    else:
        fy = fx
    if "cx" in meta:
        cx, cy = meta["cx"], meta["cy"]
    else:
        cx = 0.5 * W
        cy = 0.5 * H

    os.makedirs(os.path.join(data_dir, "blender2colmap/sparse"), exist_ok=True)

    with open(os.path.join(data_dir, "blender2colmap/sparse/cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}")

    with open(os.path.join(data_dir, "blender2colmap/sparse/points3D.txt"), "w") as f:
        f.write("")
