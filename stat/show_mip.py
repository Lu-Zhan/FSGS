import os
import json
import glob


def read_txt(txt_dir, metrics):
    txt_names = [os.path.basename(x) for x in glob.glob(os.path.join(txt_dir, "*.txt"))]
    txt_names = sorted(txt_names)

    res = {x: -1 for x in metrics}

    for txt_name in txt_names:
        r, p, s, l = txt_name[:-4].split("_")[1:]

        res[f"{r}_psnr"] = float(p)
        res[f"{r}_ssim"] = float(s)
        res[f"{r}_lpips"] = float(l)
    
    return res


def stat_mip_results(base_dir, exp_names, method_names, scene_names):
    metrics = ["psnr", "ssim", "lpips"]
    rs = ['r1', 'r2', 'r4', 'r8']
    metrics = [f"{y}_{x}" for x in metrics for y in rs]
    
    for exp_name in exp_names:

        res_dict = {}

        for method_name in method_names:
            for scene_name in scene_names:
                metric_dir = os.path.join(base_dir, exp_name, method_name, scene_name, "metrics")
                res = read_txt(metric_dir, metrics)

                if method_name not in res_dict.keys():
                    res_dict[method_name] = {scene_name: res}
                else:
                    res_dict[method_name][scene_name] = res
        
        # save res_dict as a table,
        res = []
        metrics = ["psnr", "ssim", "lpips"]
        rs = ['r1', 'r2', 'r4', 'r8']
        metrics = [f"{y}_{x}" for x in metrics for y in rs]

        for metric in metrics:
            res.append(metric.upper() + "\n")
            res.append("Method, " + ", ".join(scene_names) + "\n")
            
            for method_name in method_names:
                line = f"{method_name}, " + ", ".join([f"{res_dict[method_name][scene_name][metric]:.4f}" for scene_name in scene_names]) + "\n"
                res.append(line)
        
        csv_save_path = f"stat/results/{exp_name}_iso.csv"
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
        
        with open(csv_save_path, "w+") as f:
            f.writelines(res)


if __name__ == "__main__":
    base_dir = "/home/titan/exps/exps_frgs/few_shot"
    exp_names = ["llff_view3"]
    
    # base_dir = "/home/titan/exps/exps_frgs/byproduct"
    # exp_names = ["mip360_view12"]

    method_names = os.listdir(os.path.join(base_dir, exp_names[0]))
    method_names = sorted(method_names)

    scene_names = os.listdir(os.path.join(base_dir, exp_names[0], method_names[0]))
    scene_names = sorted(scene_names)

    # stat_json(exp_names, method_names, scene_names)
    stat_mip_results(base_dir, exp_names, method_names, scene_names)


                




