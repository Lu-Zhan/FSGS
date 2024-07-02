export CUDA_VISIBLE_DEVICES=1

dataset_dir=/home/luzhan/nerf_llff_data
exp_dir=/home/space/exps/fsgs_exps

# dataset_names=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
dataset_names=("leaves")
# dila_factors=("0.0001" "0.001" "0.01" "0.1" "0.2" "0.3" "0.4" "0.5")

# for dataset_name in ${dataset_names[@]}; do
#     exp_name=llff/fsgs

#     python train.py \
#         --source_path ${dataset_dir}/${dataset_name} \
#         --model_path ${exp_dir}/${exp_name}/${dataset_name} \
#         --eval \
#         --n_views 3 \
#         --sample_pseudo_interval 1
    
#     python render.py \
#         --source_path ${dataset_dir}/${dataset_name} \
#         --model_path ${exp_dir}/${exp_name}/${dataset_name} \
#         --iteration 10000

#     python metrics.py \
#         --source_path ${dataset_dir}/${dataset_name} \
#         --model_path ${exp_dir}/${exp_name}/${dataset_name} \
#         --iteration 10000
# done


dila_factors=("0.0001" "0.001" "0.01" "0.1")
for dataset_name in ${dataset_names[@]}; do
    for dila_factor in ${dila_factors[@]}; do
        exp_name=llff/fsgs_dila_${dila_factor}
        python train_dila.py \
            --source_path ${dataset_dir}/${dataset_name} \
            --model_path ${exp_dir}/${exp_name}/${dataset_name} \
            --eval \
            --n_views 3 \
            --sample_pseudo_interval 1 \
            --dilation_factor ${dila_factor}
        
        python render.py \
            --source_path ${dataset_dir}/${dataset_name} \
            --model_path ${exp_dir}/${exp_name}/${dataset_name} \
            --iteration 10000
        
        python metrics.py \
            --source_path ${dataset_dir}/${dataset_name} \
            --model_path ${exp_dir}/${exp_name}/${dataset_name} \
            --iteration 10000

    done
done