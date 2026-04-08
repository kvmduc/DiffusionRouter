#!/bin/bash

# Define the list of Python scripts to run
# Define pairs of context_class and target_class
PARAM_SETS=(
    "--context_class 0 --target_class 1"
    "--context_class 0 --target_class 2"
)

SETTINGS=(
    # "1_1e3_1_fixedlr5e5_refine5"
    # "1_1e3_1_fixedlr5e5_refine1"
    "1_1e3_1_fixedlr5e5_refine3"
)

MODEL_FLAGS="--num_classes 3"

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
    /scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=1 sample_scripts/vd_image_sample_general_multihop.py.py --input_dir /scratch/DataSets/edges2shoes_splitted/test/ --save_dir "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/edges_shoes_grayscale_distill_1_1e3_1_lr1e5_fixdata_refine3/sample_080000" --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/edges_shoes_grayscale_distill_1_1e3_1_lr1e5_fixdata_refine3/ema_0.9999_080000.pt" --timestep_respacing 200 --use_ddim True --num_samples 10000 --dataset_name "edges_shoes_grayscale" $MODEL_FLAGS $params
done
done