import glob
import os
from PIL import Image

dataset_dir = "/home/space/datasets/nvs/nerf_llff_data"
scenes = os.listdir(dataset_dir)

scenes = ["trex", "horns"]

for scene in scenes:
    scene_dir = os.path.join(dataset_dir, scene)

    image_paths = glob.glob(os.path.join(scene_dir, "images", "*.JPG")) + glob.glob(os.path.join(scene_dir, "images", "*.jpg"))
    image_paths.sort()

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)

        # Resize by a factor of 2
        width, height = image.size
        image = image.resize((width // 2, height // 2))

        save_path = os.path.join(scene_dir, "images_2", f"image{i:03d}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
