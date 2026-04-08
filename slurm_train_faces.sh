#!/bin/bash
#SBATCH --job-name=vd_faces
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_faces.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_faces.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=batch-short
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_80

cd /scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/

NGPU=2

export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/faces_sketch_segment_retrain/"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 128 --learn_sigma False --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --num_classes 3 --channel_mult 1,1,2,3,4 --log_interval 200"
/scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=$NGPU scripts/vd_train.py --dataset_name 'face_sketch_segment'  --data_dir '/scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/train/' --test_data_dir '/scratch/s223719687/DiffusionRouterModel/datasets/Faces_dataset/face_sketch_segment/test/' --lr_anneal_steps 200000 --save_interval 20000 --batch_size 32  $MODEL_FLAGS --class_cond True --in_channels 6 --use_fp16 True
