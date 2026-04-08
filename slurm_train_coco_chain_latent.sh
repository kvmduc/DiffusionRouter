#!/bin/bash
#SBATCH --job-name=vd_coco_multimodal
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_coco_multimodal.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_coco_multimodal.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=batch-short
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_80

cd /scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/

NGPU=2

export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal_v2/"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --log_interval 200"
/scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=$NGPU scripts/vd_train_latent.py --dataset_name 'coco_multimodal_latent_v2'  --data_dir '/scratch/DataSets/COCO_caption/coco_stuff_latent/' --test_data_dir '/scratch/DataSets/COCO_caption/coco_stuff/' --batch_size 64 $MODEL_FLAGS --save_interval 20000 --class_cond True --in_channels 8 --no_clip_denoised --decode_while_test --resume_checkpoint "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal_v2/model180000.pt"