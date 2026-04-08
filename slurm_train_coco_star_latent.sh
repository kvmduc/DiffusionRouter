#!/bin/bash
#SBATCH --job-name=vd_coco_multimodal_v1
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_coco_multimodal_v1.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_coco_multimodal_v1.err
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


export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal/"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --log_interval 200"
/scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=$NGPU scripts/vd_train_latent.py --dataset_name 'coco_multimodal'  --data_dir '/scratch/DataSets/COCO_caption/coco_stuff_latent/' --test_data_dir '/scratch/DataSets/COCO_caption/coco_stuff/' --batch_size 128 --microbatch 64 $MODEL_FLAGS --class_cond True --in_channels 8 --no_clip_denoised --decode_while_test --resume_checkpoint "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal/model190000.pt"
