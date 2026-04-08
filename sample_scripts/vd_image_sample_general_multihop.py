"""
Train a diffusion model on img datasets. (multi-hop via-domain routing)
"""
import sys
sys.path.append("/scratch/s223719687/DiffusionRouterModel/Versatile_Diffusion_images")
import os
import argparse
import torch as th
from PIL import Image
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.aligned_image_datasets import load_aligned_data
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusers import AutoencoderKL

# ---------------- utils for routing ----------------
CLASS_NAMES = {0: "color", 1: "edge", 2: "gray", 3: "depth"}
NAME_TO_IDX = {v: k for k, v in CLASS_NAMES.items()}

PAIRS_FOLDER = {
    "0,1": "color_edge", "1,0": "color_edge",
    "0,2": "color_gray", "2,0": "color_gray",
    "0,3": "color_depth","3,0": "color_depth",
    "1,2": "edge_gray",  "2,1": "edge_gray",
    "1,3": "edge_depth", "3,1": "edge_depth",
    "2,3": "depth_gray", "3,2": "depth_gray",
}

# Your modality graph (linear chain): gray(2) <-> color(0) <-> edge(1) <-> depth(3)
CHAIN = [2, 0, 1, 3]
POS = {c:i for i, c in enumerate(CHAIN)}

def _parse_via_seq(via_seq_str):
    """
    Returns:
      None            -> direct
      []              -> direct (empty)
      ["auto"]        -> auto chain route
      [ints...]       -> explicit sequence of intermediate class indices
    Accepts names ("color,edge") or indices ("0,1"). Case-insensitive.
    """
    if via_seq_str is None:
        return None
    s = via_seq_str.strip().lower()
    if s in ("none", "", "-2"):
        return None
    if s in ("auto", "-1"):
        return ["auto"]
    toks = [t.strip() for t in s.replace(" ", "").split(",") if t.strip() != ""]
    out = []
    for t in toks:
        if t.isdigit():
            v = int(t)
            if v not in CLASS_NAMES:
                raise ValueError(f"via_seq contains invalid class index: {t}")
            out.append(v)
        else:
            if t not in NAME_TO_IDX:
                raise ValueError(f"via_seq contains invalid class name: {t}")
            out.append(NAME_TO_IDX[t])
    return out

def _auto_route(src, dst):
    """Shortest path along CHAIN from src to dst (inclusive)."""
    if src == dst:
        return [src]
    i0, i1 = POS[src], POS[dst]
    step = 1 if i1 > i0 else -1
    return CHAIN[i0:i1+step:step]

def _compose_route(src, dst, via_seq):
    """
    Build node route [src, ..., dst], where via_seq is:
      None         -> [src, dst]
      ["auto"]     -> auto path along CHAIN
      [v1, v2,...] -> [src, v1, v2, ..., dst], dedup consecutive equals
    """
    if via_seq is None:
        route = [src, dst]
    elif via_seq == ["auto"]:
        route = _auto_route(src, dst)
    else:
        route = [src] + via_seq + [dst]

    # remove consecutive duplicates
    dedup = [route[0]]
    for x in route[1:]:
        if x != dedup[-1]:
            dedup.append(x)
    return dedup

# ----------------------------------------------------

def main():
    args = create_argparser().parse_args()

    context_class = args.context_class
    target_class  = args.target_class

    # decide multi-hop route
    via_seq = _parse_via_seq(args.via_seq)
    route_nodes = _compose_route(context_class, target_class, via_seq)
    # route as edges (u->v)
    route_edges = list(zip(route_nodes[:-1], route_nodes[1:]))

    # final output dir
    final_pair_key = f"{context_class},{target_class}"
    final_folder = os.path.join(
        args.save_dir, PAIRS_FOLDER[final_pair_key], CLASS_NAMES[target_class]
    )
    os.makedirs(final_folder, exist_ok=True)

    # optional per-step dirs
    step_folders = []
    if args.save_intermediate:
        for (u, v) in route_edges[:-1]:  # exclude last edge (final saved separately)
            key = f"{u},{v}"
            step_dir = os.path.join(args.save_dir, PAIRS_FOLDER[key], CLASS_NAMES[v])
            os.makedirs(step_dir, exist_ok=True)
            step_folders.append(step_dir)
    else:
        step_folders = [None] * max(0, len(route_edges) - 1)

    dist_util.setup_dist()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    vae = None
    if args.latent_space:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dist_util.dev()).eval()

    data = load_aligned_data(
        dataset_name=args.dataset_name,
        data_dir=args.input_dir,
        batch_size=args.batch_size,
        image_size=args.image_size if not args.latent_space else 256,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log(f"routing: {' -> '.join(CLASS_NAMES[n] for n in route_nodes)}")
    logger.log("sampling...")
    images = []
    data_iter = iter(data)

    # ------- helpers -------
    def _to_uint8_bhwc(x_bchw):
        x = ((x_bchw + 1) * 127.5).clamp(0, 255).to(th.uint8)  # BxCxHxW
        x = x.permute(0, 2, 3, 1).contiguous()                 # BxHxWxC
        return x

    def _decode_if_needed(t_bchw):
        if args.latent_space:
            img = vae.decode(t_bchw / 0.18215).sample
            return _to_uint8_bhwc(img)
        else:
            return _to_uint8_bhwc(t_bchw)

    def _sample_once(x_context, ctx_cls, tgt_cls):
        """
        One step: ctx_cls -> tgt_cls, conditioning on x_context.
        x_context: Bx(3 or 4)xHxW (pixel or latent)
        returns:   Bx(3 or 4)xHxW in target domain scale (pixel in [-1,1] or latent)
        """
        model_kwargs = {
            "target_class":  tgt_cls * th.ones(x_context.shape[0], dtype=th.int64, device=dist_util.dev()),
            "context_class": ctx_cls * th.ones(x_context.shape[0], dtype=th.int64, device=dist_util.dev()),
        }
        noise = th.randn_like(x_context)
        xT = th.cat([noise, x_context], dim=1).to(dist_util.dev())  # concat target-noise || context
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        out = sample_fn(
            model,
            x_context.shape,  # target shape (C_target x H x W)
            noise=xT,         # full input (C_target+C_context)
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        return out  # same shape as x_context (target channels)

    # -----------------------
    with th.no_grad():
        while len(images) * args.batch_size < args.num_samples:
            (batch, extra) = next(data_iter)
            filenames = extra["filepath"]

            # context slice from (B, 12, H, W) -> (B,3,H,W)
            x_ctx_pix = batch[:, 3*context_class:3*context_class+3].to(dist_util.dev())

            # encode to latent if needed
            if args.latent_space:
                x_cur = vae.encode(x_ctx_pix).latent_dist.sample().mul_(0.18215)  # (B,4,32,32)
            else:
                x_cur = x_ctx_pix  # (B,3,H,W) in [-1,1]
            cur_cls = context_class

            # multi-hop sampling
            for edge_idx, (u, v) in enumerate(route_edges):
                y = _sample_once(x_cur, u, v)   # u -> v

                # save intermediate steps (except final edge; handled below)
                if args.save_intermediate and edge_idx < len(route_edges) - 1:
                    mid_img = _decode_if_needed(y)
                    gathered_mid = [th.zeros_like(mid_img) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_mid, mid_img)
                    if dist.get_rank() == 0:
                        step_dir = step_folders[edge_idx]
                        for img_np, fn in zip(gathered_mid[0].cpu().numpy(), filenames):
                            Image.fromarray(img_np, 'RGB').save(
                                os.path.join(step_dir, f"{fn[:-4]}.png")
                            )
                    del mid_img, gathered_mid

                # advance
                x_cur = y
                cur_cls = v
                del y

            # final image (after last hop)
            final_img = _decode_if_needed(x_cur)
            gathered = [th.zeros_like(final_img) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, final_img)
            images.extend([g.cpu().numpy() for g in gathered])

            if dist.get_rank() == 0:
                for img_np, fn in zip(gathered[0].cpu().numpy(), filenames):
                    Image.fromarray(img_np, 'RGB').save(os.path.join(final_folder, f"{fn[:-4]}.png"))

            logger.log(f"created {len(images) * args.batch_size} samples")

            del x_ctx_pix, x_cur, final_img, gathered

        logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5000,
        batch_size=128,
        use_ddim=False,
        model_path="",
        class_cond=True,
        in_channels=6,  # (target 3 || context 3). If using latent_space, set to 8 in your config.
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="", help="Input directory containing images.")
    parser.add_argument("--save_dir",  type=str, default=None, help="Where to save")
    parser.add_argument("--dataset_name", type=str,
                        choices=["edges_shoes_grayscale","face_sketch_segment","face_sketch_segment_latent","coco_multimodal"])
    parser.add_argument("--latent_space", dest="latent_space", action="store_true", help="Use latent space")
    parser.add_argument("--context_class", type=int, required=True, help="Context class idx {0,1,2,3}")
    parser.add_argument("--target_class",  type=int, required=True, help="Target class idx {0,1,2,3}")

    # NEW: multi-hop via routing
    parser.add_argument(
        "--via_seq", type=str, default="none",
        help=(
            "Routing through shared domains. "
            "'none' or '' for direct; 'auto' for shortest path along gray(2)-color(0)-edge(1)-depth(3); "
            "or a comma list of names/ids, e.g. 'color,edge' or '0,1'."
        )
    )
    parser.add_argument("--save_intermediate", action="store_true", help="Save all intermediate hop outputs")

    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
