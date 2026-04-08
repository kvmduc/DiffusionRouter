#!/bin/bash
#SBATCH --job-name=3gpu_vd_distill_coco_v2_multimodal_refine3
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_distill_coco_v2_multimodal_refine3.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_distill_coco_v2_multimodal_refine3.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=batch-short
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_80

cd /scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/

NGPU=2

export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal_v2_780k_distill_1_0_1_lr1e5_refine3_overlap/"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --log_interval 200"
/scratch/s223719687/.conda/envs/rectflow/bin/torchrun --nproc-per-node=$NGPU scripts/vd_distill_coco_v2_latent.py --dataset_name 'coco_multimodal_latent_v2'  --data_dir '/scratch/DataSets/COCO_caption/coco_stuff_latent/' --test_data_dir '/scratch/DataSets/COCO_caption/coco_stuff/' --lr_anneal_steps 200000 --batch_size 32 $MODEL_FLAGS --class_cond True --in_channels 8 --model_path "/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/coco_multimodal_v2/ema_0.9999_780000.pt" --save_interval 20000 --no_clip_denoised --decode_while_test --lambda_values 1.0 0 1.0 --lr 1e-5 --num_refine_steps 3