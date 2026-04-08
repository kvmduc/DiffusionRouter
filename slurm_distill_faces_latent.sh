#!/bin/bash
#SBATCH --job-name=vd_faces_distill_1_0_1_lr1e5_refine3_augment
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_faces_distill_1_0_1_lr1e5_refine3_augment.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_faces_distill_1_0_1_lr1e5_refine3_augment.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=batch-short
#SBATCH --time=120:00:00
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


export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_latent_128_distill_1_0_1_fixlr1e5_refine3_augment/"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --log_interval 200"
/scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc_per_node=$NGPU --master_port=$MASTER_PORT scripts/vd_distill_latent.py --dataset_name 'face_sketch_segment' --data_dir "/scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/train" --test_data_dir "/scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/test/" --batch_size 64 $MODEL_FLAGS --class_cond True --in_channels 8 --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_latent_128/ema_0.9999_200000.pt" --no_clip_denoised --decode_while_test --lambda_values 1.0 0 1.0 --lr 1e-5 --augment --latent_space --num_refine_steps 3
