import os
import json
import glob
from collections import OrderedDict

exp_dir = '/home/titan/exps_backup/supp_exps/fewshot/mip360_view12'
exp_names = os.listdir(exp_dir)
exp_names = [x for x in exp_names]
exp_names = sorted(exp_names, key=lambda x: float(x.split('_')[-1][3:]))

data_names = os.listdir(os.path.join(exp_dir, exp_names[0]))
data_names = sorted(data_names)

all_results = OrderedDict()

for exp_name in exp_names:
    all_results[exp_name] = {}
    for data_name in data_names:
        metrics_path = os.path.join(exp_dir, exp_name, data_name, 'results.json')

        try:
            with open(metrics_path, 'r') as f:
                results = json.load(f)
            
            results = results['ours_30000']

            psnr, ssim, lpips = results['PSNR'], results['SSIM'], results['LPIPS']
        except:
            psnr, ssim, lpips = -1, -1, -1

        all_results[exp_name][data_name] = {
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips
        }

with open('eval/mip360_view12.csv', 'w+') as f:
    f.write('method,' + ','.join(data_names) + '\n')

    for metric_type in ['psnr', 'ssim', 'lpips']:
        f.write(metric_type + '\n')
        for exp_name in exp_names:
            line = exp_name

            for data_name in data_names:
                line += f',{all_results[exp_name][data_name][metric_type]}'
            
            f.write(line + '\n')

print('done!')