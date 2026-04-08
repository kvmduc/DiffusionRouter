#!/bin/bash
#SBATCH --job-name=vd_e2s
#SBATCH --output=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/out/vd_e2s.out
#SBATCH --error=/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/logs/err/vd_e2s.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=priority
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_80

cd /scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/

NGPU=1

export OPENAI_LOGDIR="/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images/log_dir/edges_shoes_grayscale_retrain"
/scratch/s223719687/anaconda3/envs/rectflow/bin/python scripts/vd_train.py --dataset_name 'edges_shoes_grayscale' --data_dir "/scratch/DataSets/edges2shoes_splitted/train/" --lr_anneal_steps 200000 --batch_size 128 --class_cond True --in_channels 6
