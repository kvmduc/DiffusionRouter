#!/bin/bash

# Define the list of Python scripts to run
# Define pairs of context_class and target_class
PARAM_SETS=(
    "--context_class 2 --target_class 1 --via_seq 0"
    "--context_class 1 --target_class 2 --via_seq 0"
)

SETTINGS=(
    "retrain"
)

MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 128 --learn_sigma False --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --num_classes 3 --channel_mult 1,1,2,3,4"

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
    /scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=1 sample_scripts/vd_image_sample_general_multihop.py_multihop.py --input_dir /scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/test --save_dir "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_retrain/sample_200000" --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_retrain/ema_0.9999_200000.pt" --save_intermediate --timestep_respacing 200 --use_ddim True --num_samples 5000 --batch_size 256 --dataset_name "face_sketch_segment" $MODEL_FLAGS $params
done
done