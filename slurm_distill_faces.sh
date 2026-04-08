#!/bin/bash
#SBATCH --job-name=vd_faces_distill_r5_clipdenoised
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_faces_distill_r5_clipdenoised.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_faces_distill_r5_clipdenoised.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=batch-short
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_80

cd /scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/

NGPU=2

export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/face_sketch_segment_distill_1_0_1_lr5e5_refine5_clipdenoised"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 128 --learn_sigma False --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --num_classes 3 --channel_mult 1,1,2,3,4 --log_interval 200"
/scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=$NGPU  scripts/vd_distill.py --dataset_name 'face_sketch_segment' --data_dir '/scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/train/' --test_data_dir '/scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/test/' --batch_size 8 --microbatch 4 --class_cond True --in_channels 6 --num_classes 3 $MODEL_FLAGS --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_retrain/ema_0.9999_200000.pt" --use_fp16 True --save_interval 20000 --lambda_values 1.0 0 1.0 --lr 5e-5 --num_refine_steps 5
