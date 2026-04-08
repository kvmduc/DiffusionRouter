"""
Train a diffusion model on img datasets.
"""
import sys
sys.path.append("/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images")
import argparse
from glob import glob
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.aligned_image_datasets import load_data, load_aligned_data
from guided_diffusion.train_util import TrainLoop
import torch.distributed as dist
import os
import wandb
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()
    
    if dist.get_rank() == 0:
        name = 'Versatile_FaceSketchSegment_x128' if args.dataset_name == 'face_sketch_segment' else 'Versatile_EdgesShoesGrayscale_x64'
        wandb.init(project="DiffusionRouterModel", name=name, config=vars(args), mode="online", id=args.wandb_run_id, resume="must" if (args.wandb_run_id is not None) else "never")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )   
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    logger.log(f"loading data from {args.data_dir}")
    data = load_data(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
    test_data = load_aligned_data(
        dataset_name=args.dataset_name,
        data_dir=args.test_data_dir if args.test_data_dir else os.path.join(args.data_dir, 'test'),
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )
    

    logger.log("training model...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        test_data=test_data,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        choices=["edges_shoes_grayscale", "face_sketch_segment", "face_sketch_segment_latent", "coco_multimodal"], 
    )
    parser.add_argument(
        "--no_clip_denoised",
        dest="clip_denoised",
        action="store_false",
        help="Clip denoised images while sampling"
    )
    parser.add_argument(
        "--test_data_dir", 
        type=str,
        default="",
        help="Path to the test data directory"
    )
    parser.add_argument(
        "--latent_space",
        dest="latent_space",
        action="store_true",
        help="Use latent space for training",
    )
    parser.add_argument(
        "--wandb_run_id",
        dest="wandb_run_id",
        type=str,
        default=None,
        help="Use wandb run ID for resuming",
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
