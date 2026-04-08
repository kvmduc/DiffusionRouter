#!/bin/bash

# Define the list of Python scripts to run
# Define pairs of context_class and target_class
PARAM_SETS=(
    "--context_class 1 --target_class 0"
    "--context_class 2 --target_class 0"
    "--context_class 3 --target_class 0"
    "--context_class 1 --target_class 2"
    "--context_class 2 --target_class 1"
    "--context_class 1 --target_class 3"
    "--context_class 3 --target_class 1"
    "--context_class 2 --target_class 3"
    "--context_class 3 --target_class 2"
    "--context_class 0 --target_class 1"
    "--context_class 0 --target_class 2"
    "--context_class 0 --target_class 3"
)

MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True"


# Loop through each script and submit it as a separate job
for params in "${PARAM_SETS[@]}"; do
    # pick a free port for this rank
    export MASTER_PORT=$(
      python3 - <<PYCODE
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PYCODE
    )
    /scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=1 --master_port=$MASTER_PORT sample_scripts/vd_image_sample_general_multihop.py.py --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 32 --in_channels 8 --batch_size 128 --input_dir /scratch/DataSets/COCO_caption/coco_stuff/ --save_dir "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal_distill_1_1e3_1_lr1e5_refine3/sample_120000" --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal_distill_1_1e3_1_lr1e5_refine3/ema_0.9999_120000.pt" --timestep_respacing 1000 --use_ddim True --dataset_name "coco_multimodal" --latent_space --clip_denoised False  $params
done