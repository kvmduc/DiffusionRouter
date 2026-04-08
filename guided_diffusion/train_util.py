import blobfile as bf
import copy
import functools
import os
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from pytorch_fid import fid_score
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from guided_diffusion.script_util import create_gaussian_diffusion
from .resample import LossAwareSampler, UniformSampler
from guided_diffusion.augment import AugmentPipe
from PIL import Image
import wandb
import sys
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
from diffusers.models import AutoencoderKL

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            test_data=None,
            latent_space=False,
            decode_while_test=False,
            clip_denoised=True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.train_data = data
        self.test_data = test_data
        self.save_dir = logger.get_dir()
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.latent_space = latent_space
        self.clip_denoised = clip_denoised
        self.decode_while_test = decode_while_test

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        
        if self.test_data is not None:
            self.test_diffusion = create_gaussian_diffusion(
                timestep_respacing="200"
            )
        else:
            self.test_diffusion = None
        
        if self.latent_space or self.decode_while_test:
            self.autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            self.autoencoder = self.autoencoder.eval()
            self.autoencoder.train = disabled_train
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder.to(dist_util.dev())
            
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # if self.step % self.save_interval == 0:
            #     # For debug
            #     self.sample_durring_train(self.test_data, self.step + self.resume_step)
            #     sys.exit(0)
            
            batch, cond = next(self.train_data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logs = logger.dumpkvs()
                
                if dist.get_rank() == 0:
                    wandb.log(logs, step=(self.step + self.resume_step))
                    
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if (self.step % self.save_interval == 0) and (self.step > 0) and self.test_data is not None:
                with th.no_grad():
                    print("Sampling during training...")
                    self.sample_durring_train(self.test_data, self.step + self.resume_step)
                dist.barrier()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            
            if self.latent_space:
                c = micro.shape[1] // 2
                micro1 = micro[:, :c, :, :]
                micro2 = micro[:, c:, :, :]
                micro1 = self.autoencoder.encode(micro1).latent_dist.sample().mul_(0.18215)
                micro2 = self.autoencoder.encode(micro2).latent_dist.sample().mul_(0.18215)
                micro = th.cat([micro1, micro2], dim=1)
                
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                raise NotImplementedError("Loss-aware sampler not supported in this version")

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("learning_rate", self.opt.param_groups[0]["lr"])

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
                
            # TODO: save scheduler of LR

        dist.barrier()
        
    def sample_durring_train(self, test_data_loader, step):
        # images = []
        batches_processed = 0
        save_dir = os.path.join(self.save_dir, f"sample_{step:06d}")
        os.makedirs(os.path.join(save_dir, 'color_gray', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color_edge', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'gray'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'gray'), exist_ok=True)

        if self.latent_space or self.decode_while_test:
            num_samples = 5000
        else:
            num_samples = 10000
        data_iter = iter(test_data_loader)

        if self.test_diffusion is None:
            print("Using diffusion model for sampling")
            sample_fn = self.diffusion.ddim_sample_loop
        else:
            print("Using test diffusion model for sampling")
            sample_fn = self.test_diffusion.ddim_sample_loop
        
        with th.no_grad():
            while batches_processed * self.batch_size < num_samples:
                (batch, extra) = next(data_iter)
                x_color = batch[:, :3]
                x_edge = batch[:, 3:6]
                x_gray = batch[:, 6:9]
                filenames = extra["filepath"]
                
                ############################ Save original images ##############################
                # Save x_color to image
                folder = 'color'
                x_color_img = ((x_color + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_color_img = x_color_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_color_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_edge to image
                folder = 'edge'
                x_edge_img = ((x_edge + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_edge_img = x_edge_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_edge_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_gray to image
                folder = 'gray'
                x_gray_img = ((x_gray + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_gray_img = x_gray_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_gray_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                ################################################################################


                x_color = x_color.to(dist_util.dev())
                x_edge = x_edge.to(dist_util.dev())
                x_gray = x_gray.to(dist_util.dev())
                if self.latent_space or self.decode_while_test:
                    x_edge = self.autoencoder.encode(x_edge).latent_dist.sample().mul_(0.18215)
                    x_gray = self.autoencoder.encode(x_gray).latent_dist.sample().mul_(0.18215)
                
                ############################### generate edge to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_e2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_e2c = self.autoencoder.decode(sample_e2c / 0.18215).sample                                #(Bx3x256x256)
                
                sample = ((sample_e2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_edge/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))        
                    
                
                ################################### generate gray to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())

                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_g2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_g2c = self.autoencoder.decode(sample_g2c / 0.18215).sample         
                
                sample = ((sample_g2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_gray/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))   
                
                batches_processed += 1
                if batches_processed % 10 == 0:
                    print(f"Processed {batches_processed} batches, {batches_processed * self.batch_size} samples generated so far.")
                    
            ###################################### calculate fid ########################################
            e2c_fid = calc_FID(
                os.path.join(save_dir, 'color_edge', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            g2c_fid = calc_FID(
                os.path.join(save_dir, 'color_gray', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            if dist.get_rank() == 0:
                wandb.log({
                    "fid_e2c": e2c_fid,
                    "fid_g2c": g2c_fid,
                }, step=(self.step + self.resume_step))
        

class DistillLoop(TrainLoop):
    def __init__(self, lambda_values, augment, num_refine_steps=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_values = lambda_values
        self.augment = augment
        self.num_refine_steps = num_refine_steps
        print("latent encoding while training:", self.latent_space)
        print("decode while testing:", self.decode_while_test)
        print("lambda values:", self.lambda_values)
        print("Augmentation:", self.augment)
        print("Number of refine steps:", self.num_refine_steps)
        augment_kwargs = {}
        augment_kwargs.update(p=0.25, xflip=1, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        self.augment_pipe = AugmentPipe(**augment_kwargs) if self.augment else None


    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        # ------------------------------------------------------------
        # Build *static* kwargs dict once; reuse for every micro‑batch
        # ------------------------------------------------------------
        domain_channels   = self.model.in_channels // 2          # concat the channel-wise
        domain_order      = ("c1", "e", "c2", "g")
        logical_domain_map = {
            "c1": "color",
            "c2": "color",
            "e":  "edge",
            "g":  "gray",
        }

        loss_recipe = dict(
            teacher_pairs=[
                dict(src="color", dst="edge", xt="e", ctx="c2"),
                dict(src="color", dst="gray", xt="g", ctx="c1"),
            ],
            kl_pairs=[
                dict(src="gray", dst="edge", xt="e", ctx="g", teacher="color->edge"),
                dict(src="edge", dst="gray", xt="g", ctx="e", teacher="color->gray"),
            ],
            reg_pairs=[
                dict(src="gray", dst="edge", xt="e", ctx="g", noise="e"),
                dict(src="edge", dst="gray", xt="g", ctx="e", noise="g"),
            ],
            revise_pairs=[
                dict(src="color", dst="edge",  xt="e",  ctx="c1", noise="e"),
                dict(src="color", dst="gray",  xt="g",  ctx="c2", noise="g"),
                dict(src="edge",  dst="color", xt="c1", ctx="e",  noise="c1"),
                dict(src="gray",  dst="color", xt="c2", ctx="g",  noise="c2"),
            ],
        )

        static_model_kwargs = dict(
            lambda_values=self.lambda_values,       # (λ₁, λ₂, λ₃)
            domain_order=domain_order,
            domain_channels=domain_channels,
            logical_domain_map=logical_domain_map,
            num_refine_steps=self.num_refine_steps,
            clip_denoised=self.clip_denoised,
            **loss_recipe,
        )
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            domain_classes = {
            "color": micro_cond["color_class"],
            "edge":  micro_cond["edge_class"],
            "gray":  micro_cond["gray_class"],
            }

            model_kwargs = {**static_model_kwargs, **micro_cond, **{"domain_classes": domain_classes}}

            if self.augment_pipe is not None:
                micro_list = []
                args = None
                with th.no_grad():
                    C = 3
                    num_domains = micro.shape[1] // C
                    for j in range(num_domains):
                        micro_domain = micro[:, j * C:(j + 1) * C, :, :]
                        if args is None:
                            micro_domain, _, args = self.augment_pipe(micro_domain)
                        else:
                            micro_domain, _, _ = self.augment_pipe.apply(micro_domain, arguments=args)
                        micro_list.append(micro_domain)
                micro = th.cat(micro_list, dim=1)

            if self.latent_space:
                micro_list = []
                with th.no_grad():
                    C = 3
                    num_domains = micro.shape[1] // C
                    for j in range(num_domains):
                        micro_domain = micro[:, j * C:(j + 1) * C, :, :]
                        micro_domain = self.autoencoder.encode(micro_domain).latent_dist.sample().mul_(0.18215)
                        micro_list.append(micro_domain)
                micro = th.cat(micro_list, dim=1)
                

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())  
                 
            compute_losses = functools.partial(
                self.diffusion.training_without_pair_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=model_kwargs,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                raise NotImplementedError("Loss-aware sampler not supported in this version")

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            
    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):      
                
            batch, cond = next(self.train_data)
            self.run_step(batch, cond)
            
            if self.step % self.log_interval == 0:
                logs = logger.dumpkvs()
                
                if dist.get_rank() == 0:
                    wandb.log(logs, step=(self.step + self.resume_step))
                    
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if (self.step % self.save_interval == 0) and (self.step > 0) and self.test_data is not None:
                with th.no_grad():
                    print("Sampling during training...")
                    self.sample_durring_train(self.test_data, self.step + self.resume_step)
                dist.barrier()
                
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def sample_durring_train(self, test_data_loader, step):
        # images = []
        batches_processed = 0
        save_dir = os.path.join(self.save_dir, f"sample_{step:06d}")
        os.makedirs(os.path.join(save_dir, 'color_gray', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color_edge', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'gray'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'gray'), exist_ok=True)
        
        if self.latent_space or self.decode_while_test:
            num_samples = 5000
        else:
            num_samples = 10000
        data_iter = iter(test_data_loader)

        if self.test_diffusion is None:
            print("Using diffusion model for sampling")
            sample_fn = self.diffusion.ddim_sample_loop
        else:
            print("Using test diffusion model for sampling")
            sample_fn = self.test_diffusion.ddim_sample_loop
        
        with th.no_grad():
            while batches_processed * self.batch_size < num_samples:
                (batch, extra) = next(data_iter)
                x_color = batch[:, :3]
                x_edge = batch[:, 3:6]
                x_gray = batch[:, 6:9]
                filenames = extra["filepath"]
                
                ############################ Save original images ##############################
                # Save x_color to image
                folder = 'color'
                x_color_img = ((x_color + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_color_img = x_color_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_color_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_edge to image
                folder = 'edge'
                x_edge_img = ((x_edge + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_edge_img = x_edge_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_edge_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_gray to image
                folder = 'gray'
                x_gray_img = ((x_gray + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_gray_img = x_gray_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_gray_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                ################################################################################


                x_color = x_color.to(dist_util.dev())
                x_edge = x_edge.to(dist_util.dev())
                x_gray = x_gray.to(dist_util.dev())
                if self.latent_space or self.decode_while_test:
                    x_edge = self.autoencoder.encode(x_edge).latent_dist.sample().mul_(0.18215)
                    x_gray = self.autoencoder.encode(x_gray).latent_dist.sample().mul_(0.18215)
                
                ############################### generate edge to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_e2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_e2c = self.autoencoder.decode(sample_e2c / 0.18215).sample                                #(Bx3x256x256)
                
                sample = ((sample_e2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_edge/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))        
                    
                
                ################################### generate gray to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())

                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_g2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_g2c = self.autoencoder.decode(sample_g2c / 0.18215).sample         
                
                sample = ((sample_g2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_gray/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))   
                
                
                ###################################### generate gray (to color) to edge ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample       
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'edge_gray/edge'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                    
                ###################################### generate edge to gray ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample      
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'edge_gray/gray'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                
                batches_processed += 1
                if batches_processed % 10 == 0:
                    print(f"Processed {batches_processed} batches, {batches_processed * self.batch_size} samples generated so far.")
                
            ###################################### calculate fid ########################################
            e2c_fid = calc_FID(
                os.path.join(save_dir, 'color_edge', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            g2c_fid = calc_FID(
                os.path.join(save_dir, 'color_gray', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            g2e_fid = calc_FID(
                os.path.join(save_dir, 'edge_gray', 'edge'),
                os.path.join(save_dir, 'edge'),
            ).item()
            
            e2g_fid = calc_FID(
                os.path.join(save_dir, 'edge_gray', 'gray'),
                os.path.join(save_dir, 'gray'),
            ).item()
            
            if dist.get_rank() == 0:
                wandb.log({
                    "fid_e2c": e2c_fid,
                    "fid_g2c": g2c_fid,
                    "fid_e2g": e2g_fid,
                    "fid_g2e": g2e_fid,
                }, step=(self.step + self.resume_step))


class DistillLoop_COCO(DistillLoop):
    def __init__(self, lambda_values, augment, num_refine_steps=0, *args, **kwargs):
        super().__init__(lambda_values, augment, num_refine_steps, *args, **kwargs)
        print("Using COCO dataset for distillation loop")
    
    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        # ------------------------------------------------------------
        # Build *static* kwargs dict once; reuse for every micro‑batch
        # ------------------------------------------------------------
        domain_channels   = self.model.in_channels // 2          # concat the channel-wise
        domain_order      = ("c1", "e", "c2", "g", "c3", "d")
        logical_domain_map = {
            "c1": "color",
            "e":  "edge",
            "c2": "color",
            "g":  "gray",
            "c3": "color",
            "d":  "depth",
            # alias logical names used in loss_recipe
            "color1": "color",
            "color2": "color",
            "color3": "color",
        }

        loss_recipe = dict(
            teacher_pairs=[
                dict(src="color2", dst="edge", xt="e", ctx="c2"),
                dict(src="color3", dst="edge", xt="e", ctx="c3"),
                dict(src="color1", dst="gray", xt="g", ctx="c1"),
                dict(src="color3", dst="gray", xt="g", ctx="c3"),
                dict(src="color1", dst="depth", xt="d", ctx="c1"),
                dict(src="color2", dst="depth", xt="d", ctx="c2"),
            ],
            kl_pairs=[
                dict(src="gray", dst="edge", xt="e", ctx="g", teacher="color2->edge"),
                dict(src="depth", dst="edge", xt="e", ctx="d", teacher="color3->edge"),
                
                dict(src="edge", dst="gray", xt="g", ctx="e", teacher="color1->gray"),
                dict(src="depth", dst="gray", xt="g", ctx="d", teacher="color3->gray"),

                dict(src="edge", dst="depth", xt="d", ctx="e", teacher="color1->depth"),
                dict(src="gray", dst="depth", xt="d", ctx="g", teacher="color2->depth"),
            ],
            reg_pairs=[
                dict(src="gray", dst="edge", xt="e", ctx="g", noise="e"),
                dict(src="depth", dst="edge", xt="e", ctx="d", noise="e"),
                dict(src="edge", dst="gray", xt="g", ctx="e", noise="g"),
                dict(src="depth", dst="gray", xt="g", ctx="d", noise="g"),
                dict(src="edge", dst="depth", xt="d", ctx="e", noise="d"),
                dict(src="gray", dst="depth", xt="d", ctx="g", noise="d"),
            ],
            revise_pairs=[
                dict(src="color1", dst="edge",  xt="e",  ctx="c1", noise="e"),
                dict(src="color2", dst="gray",  xt="g",  ctx="c2", noise="g"),
                dict(src="color3", dst="depth", xt="d",  ctx="c3", noise="d"),
                dict(src="edge",  dst="color1", xt="c1", ctx="e",  noise="c1"),
                dict(src="gray",  dst="color2", xt="c2", ctx="g",  noise="c2"),
                dict(src="depth", dst="color3", xt="c3", ctx="d",  noise="c3"),
            ],
        )

        static_model_kwargs = dict(
            lambda_values=self.lambda_values,       # (λ₁, λ₂, λ₃)
            domain_order=domain_order,
            domain_channels=domain_channels,
            logical_domain_map=logical_domain_map,
            num_refine_steps=self.num_refine_steps,
            clip_denoised=self.clip_denoised,
            **loss_recipe,
        )
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            domain_classes = {
            "color": micro_cond["color_class"],
            "edge":  micro_cond["edge_class"],
            "gray":  micro_cond["gray_class"],
            "depth": micro_cond["depth_class"],
            }

            model_kwargs = {**static_model_kwargs, **micro_cond, **{"domain_classes": domain_classes}}

            if self.latent_space:
                micro_list = []
                with th.no_grad():
                    C = 3
                    num_domains = micro.shape[1] // C
                    for j in range(num_domains):
                        micro_domain = micro[:, j * C:(j + 1) * C, :, :]
                        micro_domain = self.autoencoder.encode(micro_domain).latent_dist.sample().mul_(0.18215)
                        micro_list.append(micro_domain)
                micro = th.cat(micro_list, dim=1)
                

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            compute_losses = functools.partial(
                self.diffusion.training_without_pair_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=model_kwargs,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                raise NotImplementedError("Loss-aware sampler not supported in this version")

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)



    def sample_durring_train(self, test_data_loader, step):
        # images = []
        batches_processed = 0
        save_dir = os.path.join(self.save_dir, f"sample_{step:06d}")
        os.makedirs(os.path.join(save_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'gray'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color_gray', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color_edge', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'gray'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_depth', 'depth'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'depth_gray', 'depth'), exist_ok=True)
        
        if self.latent_space or self.decode_while_test:
            num_samples = 5000
        else:
            num_samples = 10000
        data_iter = iter(test_data_loader)

        if self.test_diffusion is None:
            print("Using diffusion model for sampling")
            sample_fn = self.diffusion.ddim_sample_loop
        else:
            print("Using test diffusion model for sampling")
            sample_fn = self.test_diffusion.ddim_sample_loop
        
        with th.no_grad():
            while batches_processed * self.batch_size < num_samples:
                (batch, extra) = next(data_iter)
                x_color = batch[:, :3]
                x_edge = batch[:, 3:6]
                x_gray = batch[:, 6:9]
                x_depth = batch[:, 9:12]
                filenames = extra["filepath"]
                
                ############################ Save original images ##############################
                # Save x_color to image
                folder = 'color'
                x_color_img = ((x_color + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_color_img = x_color_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_color_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_edge to image
                folder = 'edge'
                x_edge_img = ((x_edge + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_edge_img = x_edge_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_edge_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_gray to image
                folder = 'gray'
                x_gray_img = ((x_gray + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_gray_img = x_gray_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_gray_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_gray to image
                folder = 'depth'
                x_depth_img = ((x_depth + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_depth_img = x_depth_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_depth_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                ################################################################################


                x_color = x_color.to(dist_util.dev())
                x_edge = x_edge.to(dist_util.dev())
                x_gray = x_gray.to(dist_util.dev())
                if self.latent_space or self.decode_while_test:
                    x_edge = self.autoencoder.encode(x_edge).latent_dist.sample().mul_(0.18215)
                    x_gray = self.autoencoder.encode(x_gray).latent_dist.sample().mul_(0.18215)
                
                ############################### generate edge to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_e2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_e2c = self.autoencoder.decode(sample_e2c / 0.18215).sample                                #(Bx3x256x256)
                
                sample = ((sample_e2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_edge/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))        
                    
                
                ################################### generate gray to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())

                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_g2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_g2c = self.autoencoder.decode(sample_g2c / 0.18215).sample         
                
                sample = ((sample_g2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_gray/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))   
                
                
                ###################################### generate gray (to color) to edge ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample       
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'edge_gray/edge'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                    
                ###################################### generate edge to gray ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample      
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'edge_gray/gray'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                    
                ###################################### generate edge to depth ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 3 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample      
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'edge_depth/depth'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                    
                ###################################### generate gray to depth ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 3 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample      
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'depth_gray/depth'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                batches_processed += 1
                if batches_processed % 10 == 0:
                    print(f"Processed {batches_processed} batches, {batches_processed * self.batch_size} samples generated so far.")

            ###################################### calculate fid ########################################
            e2c_fid = calc_FID(
                os.path.join(save_dir, 'color_edge', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            g2c_fid = calc_FID(
                os.path.join(save_dir, 'color_gray', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            g2e_fid = calc_FID(
                os.path.join(save_dir, 'edge_gray', 'edge'),
                os.path.join(save_dir, 'edge'),
            ).item()
            
            e2g_fid = calc_FID(
                os.path.join(save_dir, 'edge_gray', 'gray'),
                os.path.join(save_dir, 'gray'),
            ).item()
            
            e2d_fid = calc_FID(
                os.path.join(save_dir, 'edge_depth', 'depth'),
                os.path.join(save_dir, 'depth'),
            ).item()

            g2d_fid = calc_FID(
                os.path.join(save_dir, 'depth_gray', 'depth'),
                os.path.join(save_dir, 'depth'),
            ).item()
            
            if dist.get_rank() == 0:
                wandb.log({
                    "fid_e2c": e2c_fid,
                    "fid_g2c": g2c_fid,
                    "fid_e2g": e2g_fid,
                    "fid_g2e": g2e_fid,
                    "fid_e2d": e2d_fid,
                    "fid_g2d": g2d_fid,
                }, step=(self.step + self.resume_step))
                
                
class DistillLoop_COCO_v2(DistillLoop):
    def __init__(self, lambda_values, augment, num_refine_steps=0, *args, **kwargs):
        super().__init__(lambda_values, augment, num_refine_steps, *args, **kwargs)
        print("Using COCO dataset for distillation loop")
    
    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        # ------------------------------------------------------------
        # Build *static* kwargs dict once; reuse for every micro‑batch
        # ------------------------------------------------------------
        domain_channels   = self.model.in_channels // 2          # concat the channel-wise
        domain_order      = ("g", "c", "c1", "e1", "e", "d")     # g-c ; c1-e1 ; e-d
        logical_domain_map = {
            "g":  "gray",
            "c": "color",
            "c1": "color",
            "e1":  "edge",
            "e":  "edge",
            "d":  "depth",
            # alias logical names used in loss_recipe
            "color1": "color",
            "edge1": "edge",
        }

        loss_recipe = dict(
            teacher_pairs=[
            # 2 steps sampling
                dict(src="color", dst="edge1", xt="e1", ctx="c"),       # c -> e1
                dict(src="color", dst="edge", xt="e", ctx="c"),         # c -> e
                dict(src="color1", dst="gray", xt="g", ctx="c1"),       # c1 -> g
                
                dict(src="edge1", dst="depth", xt="d", ctx="e1"),       # e1 -> d
                dict(src="edge", dst="color1", xt="c1", ctx="e"),       # e -> c1
                dict(src="edge", dst="color", xt="c", ctx="e"),         # e -> c

            # 3 steps sampling
                dict(src="color", dst="depth", xt="d", ctx="c"),         # c -> d
                dict(src="edge", dst="gray", xt="g", ctx="e"),           # e -> g
            ],
            kl_pairs=[
            # 2 steps sampling
                dict(src="gray", dst="edge1", xt="e1", ctx="g", teacher="color->edge1"),
                dict(src="gray", dst="edge", xt="e", ctx="g", teacher="color->edge"),   
                dict(src="edge1", dst="gray", xt="g", ctx="e1", teacher="color1->gray"),
                
                dict(src="color1", dst="depth", xt="d", ctx="c1", teacher="edge1->depth"),
                dict(src="depth", dst="color1", xt="c1", ctx="d", teacher="edge->color1"),
                dict(src="depth", dst="color", xt="c", ctx="d", teacher="edge->color"),
                
            # 3 steps sampling
                dict(src="gray", dst="depth", xt="d", ctx="g", teacher="color->depth"),   
                dict(src="depth", dst="gray", xt="g", ctx="d", teacher="edge->gray"),
            ],
            reg_pairs=[
                
            ],
            revise_pairs=[
                dict(src="gray", dst="color", xt="c", ctx="g", noise="c"),   
                dict(src="color", dst="gray", xt="g", ctx="c", noise="g"),
                dict(src="color1", dst="edge1",  xt="e1",  ctx="c1", noise="e1"),
                dict(src="edge1",  dst="color1", xt="c1", ctx="e1",  noise="c1"),
                dict(src="edge", dst="depth", xt="d", ctx="e", noise="d"),
                dict(src="depth", dst="edge", xt="e", ctx="d", noise="e"),
            ],
        )

        static_model_kwargs = dict(
            lambda_values=self.lambda_values,       # (λ₁, λ₂, λ₃)
            domain_order=domain_order,
            domain_channels=domain_channels,
            logical_domain_map=logical_domain_map,
            num_refine_steps=self.num_refine_steps,
            clip_denoised=self.clip_denoised,
            **loss_recipe,
        )
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            domain_classes = {
            "color": micro_cond["color_class"],
            "edge":  micro_cond["edge_class"],
            "gray":  micro_cond["gray_class"],
            "depth": micro_cond["depth_class"],
            }

            model_kwargs = {**static_model_kwargs, **micro_cond, **{"domain_classes": domain_classes}}

            if self.latent_space:
                micro_list = []
                with th.no_grad():
                    C = 3
                    num_domains = micro.shape[1] // C
                    for j in range(num_domains):
                        micro_domain = micro[:, j * C:(j + 1) * C, :, :]
                        micro_domain = self.autoencoder.encode(micro_domain).latent_dist.sample().mul_(0.18215)
                        micro_list.append(micro_domain)
                micro = th.cat(micro_list, dim=1)
                

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            compute_losses = functools.partial(
                self.diffusion.training_without_pair_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=model_kwargs,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                raise NotImplementedError("Loss-aware sampler not supported in this version")

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


    def sample_durring_train(self, test_data_loader, step):
        # images = []
        batches_processed = 0
        save_dir = os.path.join(self.save_dir, f"sample_{step:06d}")
        os.makedirs(os.path.join(save_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'gray'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color_gray', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color_depth', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'color_edge', 'color'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'edge'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_gray', 'gray'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'edge_depth', 'depth'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'depth_gray', 'depth'), exist_ok=True)
        
        if self.latent_space or self.decode_while_test:
            num_samples = 5000
        else:
            num_samples = 10000
        data_iter = iter(test_data_loader)

        if self.test_diffusion is None:
            print("Using diffusion model for sampling")
            sample_fn = self.diffusion.ddim_sample_loop
        else:
            print("Using test diffusion model for sampling")
            sample_fn = self.test_diffusion.ddim_sample_loop
        
        with th.no_grad():
            while batches_processed * self.batch_size < num_samples:
                (batch, extra) = next(data_iter)
                x_color = batch[:, :3]
                x_edge = batch[:, 3:6]
                x_gray = batch[:, 6:9]
                x_depth = batch[:, 9:12]
                filenames = extra["filepath"]
                
                ############################ Save original images ##############################
                # Save x_color to image
                folder = 'color'
                x_color_img = ((x_color + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_color_img = x_color_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_color_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_edge to image
                folder = 'edge'
                x_edge_img = ((x_edge + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_edge_img = x_edge_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_edge_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_gray to image
                folder = 'gray'
                x_gray_img = ((x_gray + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_gray_img = x_gray_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_gray_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                # Save x_gray to image
                folder = 'depth'
                x_depth_img = ((x_depth + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_depth_img = x_depth_img.permute(0, 2, 3, 1).contiguous()
                for img, filename in zip(x_depth_img.cpu().numpy(), filenames):
                    input_image = Image.fromarray(img, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                ################################################################################


                x_color = x_color.to(dist_util.dev())
                x_edge = x_edge.to(dist_util.dev())
                x_gray = x_gray.to(dist_util.dev())
                x_depth = x_depth.to(dist_util.dev())
                if self.latent_space or self.decode_while_test:
                    x_edge = self.autoencoder.encode(x_edge).latent_dist.sample().mul_(0.18215)
                    x_gray = self.autoencoder.encode(x_gray).latent_dist.sample().mul_(0.18215)
                    x_depth = self.autoencoder.encode(x_depth).latent_dist.sample().mul_(0.18215)

                ############################### generate edge to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_e2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_e2c = self.autoencoder.decode(sample_e2c / 0.18215).sample                                #(Bx3x256x256)
                
                sample = ((sample_e2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_edge/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))        
                    
                
                ################################### generate gray to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())

                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample_g2c = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                

                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample_g2c = self.autoencoder.decode(sample_g2c / 0.18215).sample         
                
                sample = ((sample_g2c + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_gray/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))   
                
                
                ###################################### generate gray to edge ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample       
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'edge_gray/edge'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                    
                ###################################### generate depth to color ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 0 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 3 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())

                x_context = x_depth
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample      
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'color_depth/color'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                    
                ###################################### generate edge to depth ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 3 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 1 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_edge
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample      
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'edge_depth/depth'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))
                    
                ###################################### generate gray to depth ########################################
                model_kwargs = {}
                model_kwargs["target_class"] = 3 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                model_kwargs["context_class"] = 2 * th.ones(x_edge.shape[0], dtype=th.int64).to(dist_util.dev())
                
                x_context = x_gray
                noise = th.randn_like(x_context)
                input = th.cat([noise, x_context], dim=1).to(dist_util.dev())
                
                sample = sample_fn(
                    self.model,
                    x_context.shape,
                    noise=input,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                del x_context, noise, input
                if self.latent_space or self.decode_while_test:
                    sample = self.autoencoder.decode(sample / 0.18215).sample      
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # images.extend([sample.cpu().numpy() for sample in gathered_samples])
                gathered_sample_img = [samples.cpu().numpy() for samples in gathered_samples]
                
                folder = 'depth_gray/depth'
                
                for sample, filename in zip(gathered_sample_img[0], filenames):
                    input_image = Image.fromarray(sample, 'RGB')
                    input_image.save(os.path.join(save_dir, folder, f"{filename[:-4]}.png"))

                batches_processed += 1
                if batches_processed % 10 == 0:
                    print(f"Processed {batches_processed} batches, {batches_processed * self.batch_size} samples generated so far.")

            ###################################### calculate fid ########################################
            e2c_fid = calc_FID(
                os.path.join(save_dir, 'color_edge', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            g2c_fid = calc_FID(
                os.path.join(save_dir, 'color_gray', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            g2e_fid = calc_FID(
                os.path.join(save_dir, 'edge_gray', 'edge'),
                os.path.join(save_dir, 'edge'),
            ).item()
            
            d2c_fid = calc_FID(
                os.path.join(save_dir, 'color_depth', 'color'),
                os.path.join(save_dir, 'color'),
            ).item()
            
            e2d_fid = calc_FID(
                os.path.join(save_dir, 'edge_depth', 'depth'),
                os.path.join(save_dir, 'depth'),
            ).item()

            g2d_fid = calc_FID(
                os.path.join(save_dir, 'depth_gray', 'depth'),
                os.path.join(save_dir, 'depth'),
            ).item()
            
            if dist.get_rank() == 0:
                wandb.log({
                    "fid_e2c": e2c_fid,
                    "fid_g2c": g2c_fid,
                    "fid_d2c": d2c_fid,
                    "fid_g2e": g2e_fid,
                    "fid_e2d": e2d_fid,
                    "fid_g2d": g2d_fid,
                }, step=(self.step + self.resume_step))


def calc_FID(gt_path, gen_path):
    fid_value = fid_score.calculate_fid_given_paths([gt_path, gen_path],
                                                    batch_size=32,
                                                    device=th.device('cuda:0'),
                                                    dims=2048)  # 2048,768,192,64
    print('FID value:', fid_value)
    
    return fid_value


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
