#!/bin/bash
#SBATCH --job-name=vd_e2s_distill_refine3
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_e2s_distill_refine3.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_e2s_distill_refine3.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=batch-short
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_80

cd /scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/

NGPU=1

# ------------------------------------------------------------------------------
# Find a free TCP port and export it for torchrun.
# This uses Python to bind to port 0 (an ephemeral port), reads back which port
# the OS assigned, then closes the socket.
# ------------------------------------------------------------------------------
export MASTER_PORT=$(
  python3 - <<PYCODE
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PYCODE
)
# MASTER_ADDR defaults to localhost; you can set it explicitly if needed:
export MASTER_ADDR=127.0.0.1


export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/edges_shoes_grayscale_distill_1_1e3_1_lr1e5_fixdata_refine3"
/scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc_per_node=$NGPU --master_port=$MASTER_PORT scripts/vd_distill.py --dataset_name 'edges_shoes_grayscale' --data_dir "/scratch/DataSets/edges2shoes_splitted/train" --test_data_dir "/scratch/DataSets/edges2shoes_splitted/test" --batch_size 32 --microbatch 16 --class_cond True --in_channels 6 --num_classes 3 --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/edges_shoes_grayscale_retrain/ema_0.9999_160000.pt" --save_interval 10000 --lambda_values 1.0 1e-3 1.0 --lr 1e-5 --num_refine_steps 3
