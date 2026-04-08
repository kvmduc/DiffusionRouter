#!/bin/bash

# Define the list of Python scripts to run
# Define pairs of context_class and target_class
PARAM_SETS=(
    "--context_class 0 --target_class 1"
    "--context_class 0 --target_class 2"
)

SETTINGS=(
    "1_1e3_1_fixedlr5e5_refine3"
)

MODEL_FLAGS="--diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --channel_mult 1,1,2,2,4,4 --num_channels 64 --attention_resolutions 16,32 --num_head_channels 8"

for settings in "${SETTINGS[@]}"; do
for params in "${PARAM_SETS[@]}"; do

    echo "Running sample script with settings: $settings and params: $params"

    # pick a free port for this rank
    export MASTER_PORT=$(
    python3 - <<PYCODE
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PYCODE
    )
    /scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=1 sample_scripts/vd_image_sample_general_multihop.py.py --diffusion_steps 1000 --image_size 32  --channel_mult 1,1,2,2,4,4 --num_channels 64 --attention_resolutions 16,32 --num_head_channels 8 --in_channels 8 --batch_size 64 --input_dir /scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/test --save_dir "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_latent_distill_small_$settings/sample_100000" --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_latent_distill_small_$settings/ema_0.9999_100000.pt" --timestep_respacing 200 --use_ddim True --dataset_name "face_sketch_segment" --latent_space --clip_denoised False $MODEL_FLAGS  $params
done
done