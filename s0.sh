dataset_dir=/home/luzhan/nerf_llff_data/fern
dataset_name=fern
exp_dir=/home/space/exps/fsgs_exps
exp_name=llff/fsgs

python train.py \
    --source_path ${dataset_dir}/${dataset_name} \
    --model_path ${exp_dir}/${exp_name}/${dataset_name} \
    --eval \
    --n_views 3 \
    --sample_pseudo_interval 1

exp_name=llff/fsgs_dila
python train_dila.py \
    --source_path ${dataset_dir}/${dataset_name} \
    --model_path ${exp_dir}/${exp_name}/${dataset_name} \
    --eval \
    --n_views 3 \
    --sample_pseudo_interval 1