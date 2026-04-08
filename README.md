# DiffusionRouter

Implementation of **"Universal Multi-Domain Translation via Diffusion Routers"** — a framework for cross-domain image translation using conditional diffusion models with multi-hop routing.

## Overview

DiffusionRouter learns to translate images between visual domains (color, edge, grayscale, depth, sketch, segmentation) by training conditional diffusion models on aligned image pairs. The key innovation is **routing**: rather than translating directly between distant domains, the model chains intermediate hops to reduce noise accumulation and improve quality.

**Supported domain sets:**
| Dataset | Domains |
|---|---|
| Edges & Shoes | edge sketches ↔ color ↔ grayscale |
| Faces | face photos ↔ sketches ↔ segmentation maps |
| COCO | color ↔ edge ↔ grayscale ↔ depth |

## Architecture

- **Backbone**: Class-conditional UNet diffusion model
- **Sampling**: DDIM with configurable respacing (default 200 steps)
- **Latent space**: Optional Stable Diffusion VAE encoding for 4–8× efficiency
- **Topologies**: Star (hub-and-spoke) and chain for COCO multi-domain routing
- **Distillation**: Knowledge distillation from teacher to student models with weighted L1 + reconstruction + KL loss

```
Training → Distillation → Multi-hop Sampling
   ↓              ↓               ↓
vd_train.py  vd_distill.py  vd_image_sample_general_multihop.py
```

## Repository Structure

```
DiffusionRouter/
├── guided_diffusion/               # Core library
│   ├── unet.py                     # UNet model architecture
│   ├── gaussian_diffusion.py       # Diffusion process (DDPM/DDIM)
│   ├── train_util.py               # Training and distillation loops
│   ├── aligned_image_datasets.py   # Datasets for training
│   ├── distill_image_datasets.py   # Datasets for distillation
│   ├── script_util.py              # Model factory & argument parsing
│   ├── respace.py                  # Timestep respacing for fast sampling
│   └── ...                         # FP16, logging, augmentation, etc.
├── scripts/                        # Entry points
│   ├── vd_train.py                 # Pixel-space training
│   ├── vd_train_latent.py          # Latent-space training
│   ├── vd_distill.py               # Distillation (pixel)
│   ├── vd_distill_latent.py        # Distillation (latent)
│   ├── vd_distill_coco_star_latent.py  # COCO star topology
│   └── vd_distill_coco_chain_latent.py # COCO chain topology
├── sample_scripts/
│   └── vd_image_sample_general_multihop.py  # Multi-hop inference
├── bash_*.sh                       # Local sampling scripts
└── slurm_*.sh                      # SLURM cluster job scripts
```

## Usage

### 1. Training

**Pixel space (Edges & Shoes):**
```bash
python scripts/vd_train.py \
  --dataset_name edges_shoes_grayscale \
  --data_dir /path/to/edges2shoes/train/ \
  --lr_anneal_steps 200000 \
  --batch_size 128 \
  --class_cond True \
  --in_channels 6
```

**Latent space (Faces):**
```bash
python scripts/vd_train_latent.py \
  --dataset_name face_sketch_segment_latent \
  --data_dir /path/to/faces/train/ \
  --lr_anneal_steps 200000 \
  --batch_size 64 \
  --class_cond True \
  --in_channels 8
```

**COCO (latent, star topology):**
```bash
torchrun --nproc-per-node=2 scripts/vd_train_latent.py \
  --dataset_name coco_multimodal_v2 \
  --data_dir /path/to/coco_stuff/ \
  --image_size 32 \
  --in_channels 8
```

Set `OPENAI_LOGDIR` to control where checkpoints are saved.

### 2. Distillation

Distill a pre-trained teacher model to a student:

```bash
python scripts/vd_distill.py \
  --dataset_name edges_shoes_grayscale \
  --data_dir /path/to/data/ \
  --model_path /path/to/teacher.pt \
  --lambda_values 1,1000,1 \
  --refine_steps 3
```

For latent COCO with chain topology:
```bash
torchrun --nproc-per-node=2 scripts/vd_distill_coco_chain_latent.py \
  --dataset_name coco_multimodal \
  --data_dir /path/to/coco_stuff/ \
  --model_path /path/to/teacher.pt
```

### 3. Sampling (Multi-hop)

Translate an image between domains, optionally routing through intermediate steps:

```bash
torchrun --nproc-per-node=1 sample_scripts/vd_image_sample_general_multihop.py \
  --model_path /path/to/model.pt \
  --input_dir /path/to/input/ \
  --save_dir /path/to/output/ \
  --context_class 1 \   # source domain (1 = edge)
  --target_class 0 \    # target domain (0 = color)
  --dataset_name coco_multimodal \
  --latent_space \
  --use_ddim True \
  --timestep_respacing 1000
```

**COCO domain class IDs:** `0=color`, `1=edge`, `2=grayscale`, `3=depth`

Use `--via_seq` to specify explicit intermediate hops, or let the model auto-route along the predefined chain `[gray → color → edge → depth]`.

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--num_channels` | 128 | UNet base channels |
| `--num_res_blocks` | 2 | Residual blocks per resolution |
| `--attention_resolutions` | 32,16,8 | Resolutions with attention |
| `--diffusion_steps` | 1000 | Training timesteps |
| `--image_size` | 32/64/128 | Spatial resolution |
| `--in_channels` | 6/8 | Input channels (pixel/latent) |
| `--ema_rate` | 0.9999 | EMA decay rate |
| `--lr` | 1e-4 | Learning rate |

## SLURM Cluster

Pre-configured SLURM scripts are provided for H100 GPU clusters. Adjust paths and partition names as needed:

```bash
sbatch slurm_train_shoes.sh           # Train edges-shoes-grayscale
sbatch slurm_train_faces_latent.sh    # Train faces (latent)
sbatch slurm_train_coco_star_latent.sh  # Train COCO star topology

sbatch slurm_distill_shoes.sh         # Distill shoes model
sbatch slurm_distill_coco_chain_latent.sh  # Distill COCO chain
```

## Requirements

```
torch
torchvision
diffusers          # Stable Diffusion VAE
wandb              # Experiment tracking
mpi4py
scikit-image
Pillow
blobfile
```

## Citation

If you use this code, please cite the corresponding paper on universal multi-domain translation via diffusion routers.
