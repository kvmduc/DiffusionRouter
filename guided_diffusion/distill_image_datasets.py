import math
import os
import random

import blobfile as bf
import numpy as np
from PIL import Image
from guided_diffusion.aligned_image_datasets import _list_image_files_recursively, _list_tensor_files_recursively
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset, Subset
import torch
from pathlib import Path


def load_distill_data(
        *,
        dataset_name,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=False,
        in_channels=3,
        return_filenames=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param in_channels: new parameter in DDIBs as we experimented with grayscale
                        images
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if dataset_name == 'edges_shoes_grayscale':
        
        edge_paths = _list_image_files_recursively(os.path.join(data_dir, 'A'))
        color_paths = _list_image_files_recursively(os.path.join(data_dir, 'B'))
        gray_paths = _list_image_files_recursively(os.path.join(data_dir, 'grayscale_rot20'))
    
        dataset = Edges_Shoes_Grayscale(
            image_size,
            edge_paths,
            color_paths,
            gray_paths,
            classes=None,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels,
            return_filenames=return_filenames,
        )
    
    elif dataset_name == 'face_sketch_segment':
        face_sketch_dir = os.path.join(data_dir, 'face_sketch')
        face_segment_dir = os.path.join(data_dir, 'face_segment')
        
        dataset = Face_Sketch_Segment(
            image_size,
            face_sketch_dir,
            face_segment_dir,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels
        )
    elif dataset_name == 'face_sketch_segment_latent':
        face_sketch_dir = os.path.join(data_dir, 'face_sketch')
        face_segment_dir = os.path.join(data_dir, 'face_segment')
        
        dataset = Face_Sketch_Segment_latent(
            None,
            face_sketch_dir,
            face_segment_dir,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels
        )
        
    elif dataset_name == 'coco_multimodal_latent':
        root_dir = data_dir
        dataset = COCO_multimodal_partial_latent(
            None,
            root_dir,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels
        )
    elif dataset_name == 'coco_multimodal_latent_v2':
        root_dir = data_dir
        dataset = COCO_multimodal_partial_latent_v2(
            None,
            root_dir,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
        
        

class Edges_Shoes_Grayscale(Dataset):
    def __init__(
            self,
            resolution,
            edge_paths,
            color_paths,
            gray_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            in_channels=3,
            filepaths=None,
            return_filenames=False,
    ):
        super().__init__()
        self.resolution = resolution
        edge_paths = edge_paths[shard:][::num_shards]
        color_paths = color_paths[shard:][::num_shards]
        gray_paths = gray_paths[shard:][::num_shards]
        self.local_classes = None
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.filepaths = filepaths
        self.return_filenames = return_filenames

        self.percent_overlap = 0.0
        
        self.num_overlap_samples = int(len(color_paths) * self.percent_overlap)  
        self.remaining_samples = int(len(color_paths) - self.num_overlap_samples)
        
        # Pair 1 
        self.to_idx_pair_1 = self.num_overlap_samples + int(self.remaining_samples // 2)
        self.edge_paths = edge_paths[:self.num_overlap_samples] + edge_paths[self.num_overlap_samples:self.to_idx_pair_1]
        self.color1_paths = color_paths[:self.num_overlap_samples] + color_paths[self.num_overlap_samples:self.to_idx_pair_1]
        
        # Pair 2
        self.gray_paths = gray_paths[:self.num_overlap_samples] + gray_paths[self.to_idx_pair_1:]
        self.color2_paths = color_paths[:self.num_overlap_samples] + color_paths[self.to_idx_pair_1:]
        
        assert len(self.edge_paths) == len(self.color1_paths) 
        assert len(self.gray_paths) == len(self.color2_paths)
        
        """
        class index:
        0: x_color                      common domain
        1: x_edge
        2: x_grayscale
        """

    
    def __len__(self):
        return int(max(len(self.color1_paths), len(self.color2_paths)))
    
    def __getitem__(self, idx):
        
        color1_path = self.color1_paths[idx % len(self.color1_paths)]
        edge_path = self.edge_paths[idx % len(self.edge_paths)]
        color2_path = self.color2_paths[idx % len(self.color2_paths)]
        gray_path = self.gray_paths[idx % len(self.gray_paths)]
        
        with bf.BlobFile(color1_path, "rb") as f:
            color1_img = Image.open(f)
            color1_img.load()
            
        with bf.BlobFile(edge_path, "rb") as f:
            edge_img = Image.open(f)
            edge_img.load()
            
        with bf.BlobFile(color2_path, "rb") as f:
            color2_img = Image.open(f)
            color2_img.load()
            
        with bf.BlobFile(gray_path, "rb") as f:
            gray_img = Image.open(f)
            gray_img.load()
            
        if self.in_channels == 1:
            color1_img = color1_img.convert("L")
            edge_img = edge_img.convert("L")
            color2_img = color2_img.convert("L")
            gray_img = gray_img.convert("L")
        elif self.in_channels == 3:
            color1_img = color1_img.convert("RGB")
            edge_img = edge_img.convert("RGB")
            color2_img = color2_img.convert("RGB")
            gray_img = gray_img.convert("RGB")

        color1_img = color1_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        color1_img = np.array(color1_img)
        edge_img = edge_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        edge_img = np.array(edge_img)
        color2_img = color2_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        color2_img = np.array(color2_img)
        gray_img = gray_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        gray_img = np.array(gray_img)
                    

        if len(color1_img.shape) < 3:
            color1_img = color1_img[:, :, np.newaxis]  # Adds a single channel
        if len(edge_img.shape) < 3:
            edge_img = edge_img[:, :, np.newaxis]
        if len(color2_img.shape) < 3:
            color2_img = color2_img[:, :, np.newaxis]
        if len(gray_img.shape) < 3:
            gray_img = gray_img[:, :, np.newaxis]
            
        if self.random_flip and random.random() < 0.5:
            color1_img = color1_img[:, ::-1]
            edge_img = edge_img[:, ::-1]
            color2_img = color2_img[:, ::-1]
            gray_img = gray_img[:, ::-1]
            
            
        color1_img = color1_img.astype(np.float32) / 127.5 - 1
        edge_img = edge_img.astype(np.float32) / 127.5 - 1
        color2_img = color2_img.astype(np.float32) / 127.5 - 1
        gray_img = gray_img.astype(np.float32) / 127.5 - 1
            
        color1_img = np.transpose(color1_img, (2, 0, 1))  # Convert to CHW format
        edge_img = np.transpose(edge_img, (2, 0, 1))
        color2_img = np.transpose(color2_img, (2, 0, 1))
        gray_img = np.transpose(gray_img, (2, 0, 1))
        
        out_dict = dict()
        out_dict["color_class"] = np.array(0, dtype=np.int64)
        out_dict["edge_class"] = np.array(1, dtype=np.int64)
        out_dict["gray_class"] = np.array(2, dtype=np.int64)
        
        return np.vstack((color1_img, edge_img, color2_img, gray_img), dtype=np.float32), out_dict
    
    
    

class Face_Sketch_Segment(Dataset):
    def __init__(
            self,
            resolution,
            face_sketch_dir,
            face_segment_dir,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            in_channels=3,
            filepaths=None,
            return_filenames=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.filepaths = filepaths
        self.return_filenames = return_filenames


        self.face_sketch_dir = face_sketch_dir
        self.face_segment_dir = face_segment_dir
        
        # Pair 1 
        self.image_paths_face1 = _list_image_files_recursively(os.path.join(self.face_sketch_dir, 'face'))       # pair with sketch
        self.image_paths_sketch = _list_image_files_recursively(os.path.join(self.face_sketch_dir, 'sketch'))
        
        # Pair 2
        self.image_paths_face2 = _list_image_files_recursively(os.path.join(self.face_segment_dir, 'face'))      # pair with segment
        self.image_paths_segment = _list_image_files_recursively(os.path.join(self.face_segment_dir, 'segment'))

        assert len(self.image_paths_face1) == len(self.image_paths_sketch)
        assert len(self.image_paths_face2) == len(self.image_paths_segment)        

        """
        class index:
        0: x_color                      common domain
        1: x_sketch
        2: x_segment
        """
        
    
    def __len__(self):
        return int(max(len(self.image_paths_face1), len(self.image_paths_face2)))
    
    def __getitem__(self, idx):

        color1_path = self.image_paths_face1[idx % len(self.image_paths_face1)]
        edge_path = self.image_paths_sketch[idx % len(self.image_paths_sketch)]
        
        # Randomly select the second pair
        idx2 = random.randrange(len(self))                      # randomly select the second pair
        
        color2_path = self.image_paths_face2[idx2 % len(self.image_paths_face2)]
        gray_path = self.image_paths_segment[idx2 % len(self.image_paths_segment)]

        with bf.BlobFile(color1_path, "rb") as f:
            color1_img = Image.open(f)
            color1_img.load()
            
        with bf.BlobFile(edge_path, "rb") as f:
            edge_img = Image.open(f)
            edge_img.load()
            
        with bf.BlobFile(color2_path, "rb") as f:
            color2_img = Image.open(f)
            color2_img.load()
            
        with bf.BlobFile(gray_path, "rb") as f:
            gray_img = Image.open(f)
            gray_img.load()
            
        if self.in_channels == 1:
            color1_img = color1_img.convert("L")
            edge_img = edge_img.convert("L")
            color2_img = color2_img.convert("L")
            gray_img = gray_img.convert("L")
        elif self.in_channels == 3:
            color1_img = color1_img.convert("RGB")
            edge_img = edge_img.convert("RGB")
            color2_img = color2_img.convert("RGB")
            gray_img = gray_img.convert("RGB")

        color1_img = color1_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        color1_img = np.array(color1_img)
        edge_img = edge_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        edge_img = np.array(edge_img)
        color2_img = color2_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        color2_img = np.array(color2_img)
        gray_img = gray_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        gray_img = np.array(gray_img)
                    

        if len(color1_img.shape) < 3:
            color1_img = color1_img[:, :, np.newaxis]  # Adds a single channel
        if len(edge_img.shape) < 3:
            edge_img = edge_img[:, :, np.newaxis]
        if len(color2_img.shape) < 3:
            color2_img = color2_img[:, :, np.newaxis]
        if len(gray_img.shape) < 3:
            gray_img = gray_img[:, :, np.newaxis]
            
        if self.random_flip and random.random() < 0.5:
            color1_img = color1_img[:, ::-1]
            edge_img = edge_img[:, ::-1]
            color2_img = color2_img[:, ::-1]
            gray_img = gray_img[:, ::-1]
            
            
        color1_img = color1_img.astype(np.float32) / 127.5 - 1
        edge_img = edge_img.astype(np.float32) / 127.5 - 1
        color2_img = color2_img.astype(np.float32) / 127.5 - 1
        gray_img = gray_img.astype(np.float32) / 127.5 - 1
            
        color1_img = np.transpose(color1_img, (2, 0, 1))  # Convert to CHW format
        edge_img = np.transpose(edge_img, (2, 0, 1))
        color2_img = np.transpose(color2_img, (2, 0, 1))
        gray_img = np.transpose(gray_img, (2, 0, 1))
        
        out_dict = dict()
        out_dict["color_class"] = np.array(0, dtype=np.int64)
        out_dict["edge_class"] = np.array(1, dtype=np.int64)
        out_dict["gray_class"] = np.array(2, dtype=np.int64)
        
        return np.vstack((color1_img, edge_img, color2_img, gray_img), dtype=np.float32), out_dict
    
    
class Face_Sketch_Segment_latent(Dataset):
    def __init__(
            self,
            resolution,
            face_sketch_dir,
            face_segment_dir,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            in_channels=3,
            filepaths=None,
            return_filenames=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.filepaths = filepaths
        self.return_filenames = return_filenames
        self.scale_factor = 0.18215


        self.face_sketch_dir = face_sketch_dir
        self.face_segment_dir = face_segment_dir
        
        # Pair 1 
        self.image_paths_face1 = _list_tensor_files_recursively(os.path.join(self.face_sketch_dir, 'face'))       # pair with sketch
        self.image_paths_sketch = _list_tensor_files_recursively(os.path.join(self.face_sketch_dir, 'sketch'))
        
        # Pair 2
        self.image_paths_face2 = _list_tensor_files_recursively(os.path.join(self.face_segment_dir, 'face'))      # pair with segment
        self.image_paths_segment = _list_tensor_files_recursively(os.path.join(self.face_segment_dir, 'segment'))
        
        # Train with 2 flow, forward and backward pair
        self.target_data = self.image_paths_face1 + self.image_paths_face2 + self.image_paths_sketch + self.image_paths_segment
        self.context_data = self.image_paths_sketch + self.image_paths_segment + self.image_paths_face1 + self.image_paths_face2

        assert len(self.image_paths_face1) == len(self.image_paths_sketch)
        assert len(self.image_paths_face2) == len(self.image_paths_segment)        

        """
        class index:
        0: x_color                      common domain
        1: x_sketch
        2: x_segment
        """
        self.target_classes = [0] * len(self.image_paths_face1) + [0] * len(self.image_paths_face2) + [1] * len(self.image_paths_sketch) + [2] * len(self.image_paths_segment)
        self.context_classes = [1] * len(self.image_paths_sketch) + [2] * len(self.image_paths_segment) + [0] * len(self.image_paths_face1) + [0] * len(self.image_paths_face2)
        
        assert len(self.target_classes) == len(self.context_classes) 
        
    
    def __len__(self):
        return len(self.target_data)
    
    def _sample_moments(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        if z.dim() == 4:
            z.squeeze_(0)  # Remove the batch dimension if it exists
        return z

    def __getitem__(self, idx):
        
        
        f1 = self.image_paths_face1[idx % len(self.image_paths_face1)]
        sk = self.image_paths_sketch[idx % len(self.image_paths_sketch)]
        
        idx2 = random.randrange(len(self))                      # randomly select the second pair
        
        f2 = self.image_paths_face2[idx2 % len(self.image_paths_face2)]
        seg = self.image_paths_segment[idx2 % len(self.image_paths_segment)]
        
        # load moments
        def load_path(p):
            with bf.BlobFile(p, "rb") as f:
                return torch.load(f, map_location="cpu")
        m1 = self._sample_moments(load_path(f1))
        m2 = self._sample_moments(load_path(sk))
        m3 = self._sample_moments(load_path(f2))
        m4 = self._sample_moments(load_path(seg))

        # stack into (4*C, H, W)
        out_tensor = torch.cat((m1, m2, m3, m4), dim=0).numpy().astype(np.float32)

        out_dict = dict()
        out_dict["color_class"] = np.array(0, dtype=np.int64)
        out_dict["edge_class"] = np.array(1, dtype=np.int64)
        out_dict["gray_class"] = np.array(2, dtype=np.int64)

        return out_tensor, out_dict
    
    

class COCO_multimodal_latent(Dataset):
    def __init__(
            self,
            resolution,
            root_dir,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            in_channels=3,
            filepaths=None,
            return_filenames=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.filepaths = filepaths
        self.return_filenames = return_filenames
        self.scale_factor = 0.18215


        self.root_dir = root_dir
        stage = 'train'
        
        # Fully path of datasets
        self.image_dir = os.path.join(self.root_dir, 'images', f'{stage}2017')
        self.sketch_dir = os.path.join(self.root_dir, 'edges', f'{stage}2017')
        self.segmentation_dir = os.path.join(self.root_dir, 'annotations', f'{stage}2017')
        self.depthmap_dir = os.path.join(self.root_dir, 'depthmaps', f'{stage}2017')


        self.image_paths = _list_tensor_files_recursively(self.image_dir)
        self.sketch_paths = [os.path.join(self.sketch_dir, Path(p).stem + '.pt') for p in self.image_paths]
        self.segmentation_paths = [os.path.join(self.segmentation_dir, Path(p).stem + '.pt') for p in self.image_paths]
        self.depthmap_paths = [os.path.join(self.depthmap_dir, Path(p).stem + '-dpt_beit_large_512.pt') for p in self.image_paths]

        assert len(self.image_paths) == len(self.sketch_paths) == len(self.segmentation_paths) == len(self.depthmap_paths), \
            "Every image must have a sketch, segmentation & depth latent file."
        
    
    def __len__(self):
        return len(self.image_paths)
    
    def _sample_moments(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        if z.dim() == 4:
            z.squeeze_(0)  # Remove the batch dimension if it exists
        return z

    def __getitem__(self, idx):
        c = self.image_paths[idx % len(self.image_paths)]
        sk = self.sketch_paths[idx % len(self.sketch_paths)]
        seg = self.segmentation_paths[idx % len(self.segmentation_paths)]
        d = self.depthmap_paths[idx % len(self.depthmap_paths)]
        
        
        # load moments
        def load_path(p):
            with bf.BlobFile(p, "rb") as f:
                return torch.load(f, map_location="cpu")
        m1 = self._sample_moments(load_path(c))
        m2 = self._sample_moments(load_path(sk))
        m3 = self._sample_moments(load_path(seg))
        m4 = self._sample_moments(load_path(d))

        # stack into (4*C, H, W)
        out_tensor = torch.cat((m1, m2, m3, m4), dim=0).numpy().astype(np.float32)

        out_dict = dict()
        out_dict["color_class"] = np.array(0, dtype=np.int64)
        out_dict["edge_class"] = np.array(1, dtype=np.int64)
        out_dict["gray_class"] = np.array(2, dtype=np.int64)
        out_dict["depth_class"] = np.array(3, dtype=np.int64)

        return out_tensor, out_dict
    
    

class COCO_multimodal_partial_latent(Dataset):
    """
    Deterministic three-pair dataset (latents), robust to missing modalities.
          <-> Seg
    Color <-> Edge  
          <-> Depth
    Pairs: (color↔edge), (color↔seg), (color↔depth).

    Parameters
    ----------
    full_overlap_target : int
        Desired overlap shared by ALL THREE pairs (default 45k). Will be clamped
        to the size of the triple-intersection actually available.
    extra_per_pair_target : int
        Desired extra anchors per pair beyond the shared overlap (default 25k),
        applied independently to each pair. If a pair has fewer available IDs,
        it is reduced (not padded with noise).

    __len__()
        By default returns the maximum of the three pair lengths so you can still
        iterate ~70k steps even if the depth pair is smaller; shorter pairs are
        cycled with modulo indexing (deterministic).
    """
    def __init__(
        self,
        resolution,
        root_dir,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        full_overlap_target=45_000,
        extra_per_pair_target=25_000,
        in_channels=3,
        return_filenames=False,
        scale_factor=0.18215,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.return_filenames = return_filenames
        self.scale_factor = scale_factor

        stage = "train"
        self.image_dir        = os.path.join(root_dir, "images",      f"{stage}2017")
        self.sketch_dir       = os.path.join(root_dir, "edges",       f"{stage}2017")
        self.segmentation_dir = os.path.join(root_dir, "annotations", f"{stage}2017")
        self.depthmap_dir     = os.path.join(root_dir, "depthmaps",   f"{stage}2017")

        # Anchor by the color list (sorted, deterministic)
        self.image_paths = _list_tensor_files_recursively(self.image_dir)
        N = len(self.image_paths)
        if N == 0:
            raise ValueError("No color latent .pt files found.")

        # Derived paths (may not exist for some IDs)
        self.sketch_paths = [os.path.join(self.sketch_dir,      Path(p).stem + ".pt")
                             for p in self.image_paths]
        self.seg_paths    = [os.path.join(self.segmentation_dir, Path(p).stem + ".pt")
                             for p in self.image_paths]
        self.depth_paths  = [os.path.join(self.depthmap_dir,    Path(p).stem + "-dpt_beit_large_512.pt")
                             for p in self.image_paths]

        # Availability masks (deterministic: scan in sorted color order)
        ids = np.arange(N, dtype=np.int64)
        ids_ce = np.array([i for i in ids], dtype=np.int64)
        ids_cg = np.array([i for i in ids],    dtype=np.int64)
        ids_cd = np.array([i for i in ids],  dtype=np.int64)

        if len(ids_cd) == 0:
            raise ValueError("No color↔depth pairs available. Depth set is empty.")

        # Triple-intersection for shared overlap
        ids_triple = np.intersect1d(np.intersect1d(ids_ce, ids_cg), ids_cd, assume_unique=False)
        self.full_overlap = min(int(full_overlap_target), int(len(ids_triple)))
        overlap_ids = ids_triple[:self.full_overlap]  # deterministic prefix


        # Extras per pair: from their available sets excluding overlap AND
        # excluding extras already taken by previous pairs (to make them disjoint).
        def pick_extras(available_ids: np.ndarray, forbidden: np.ndarray, k_target: int) -> np.ndarray:
            rem = np.setdiff1d(available_ids, forbidden, assume_unique=False)
            k = min(int(k_target), int(len(rem)))
            return rem[:k]  # deterministic prefix
        
        used = overlap_ids  # start with the shared overlap as forbidden
        extra_ce = pick_extras(ids_ce, used, extra_per_pair_target)
        used = np.union1d(used, extra_ce)

        extra_cg = pick_extras(ids_cg, used, extra_per_pair_target)
        used = np.union1d(used, extra_cg)

        extra_cd = pick_extras(ids_cd, used, extra_per_pair_target)  # reduced if depth is scarce


        # Final per-pair anchor arrays (possibly different lengths!)
        self.anchor_ce = np.concatenate([overlap_ids, extra_ce], axis=0)
        self.anchor_cg = np.concatenate([overlap_ids, extra_cg], axis=0)
        self.anchor_cd = np.concatenate([overlap_ids, extra_cd], axis=0)

        # Keep lengths around
        self.len_ce = len(self.anchor_ce)
        self.len_cg = len(self.anchor_cg)
        self.len_cd = len(self.anchor_cd)

        if self.len_ce == 0 or self.len_cg == 0 or self.len_cd == 0:
            raise ValueError(
                f"Empty pair set(s): CE={self.len_ce}, CG={self.len_cg}, CD={self.len_cd}. "
                "Check your data availability."
            )

        # Dataset length = max so we can still use larger pairs fully.
        self.length = max(self.len_ce, self.len_cg, self.len_cd)

    def __len__(self):
        return self.length

    # ----- latent helpers -----
    def _load_pt(self, p):
        with bf.BlobFile(p, "rb") as f:
            return torch.load(f, map_location="cpu")

    def _sample_moments(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        if z.dim() == 4:
            z.squeeze_(0)
        return z

    def __getitem__(self, idx):
        
        
        if idx < self.full_overlap:
            # First part of the dataset is the full-overlap set
            idx1 = idx2 = idx3 = idx
        else:        
            # Map idx to each pair deterministically with modulo (cycles shorter pairs)
            idx1 = random.randint(0, self.len_ce - 1)
            idx2 = random.randint(0, self.len_cg - 1)
            idx3 = random.randint(0, self.len_cd - 1)

        i_ce = self.anchor_ce[idx1]
        i_cg = self.anchor_cg[idx2]
        i_cd = self.anchor_cd[idx3]

        # --- color↔edge ---
        z_c_e = self._sample_moments(self._load_pt(self.image_paths[i_ce]))
        z_e   = self._sample_moments(self._load_pt(self.sketch_paths[i_ce]))

        # --- color↔seg ---
        z_c_g = self._sample_moments(self._load_pt(self.image_paths[i_cg]))
        z_g   = self._sample_moments(self._load_pt(self.seg_paths[i_cg]))

        # --- color↔depth ---
        z_c_d = self._sample_moments(self._load_pt(self.image_paths[i_cd]))
        z_d   = self._sample_moments(self._load_pt(self.depth_paths[i_cd]))

        out = torch.cat([z_c_e, z_e, z_c_g, z_g, z_c_d, z_d], dim=0).numpy().astype(np.float32)

    
        out_dict = {
            "color_class": np.array(0, dtype=np.int64),
            "edge_class":  np.array(1, dtype=np.int64),
            "gray_class":  np.array(2, dtype=np.int64),
            "depth_class": np.array(3, dtype=np.int64),
        }
        return out, out_dict
    
    
class COCO_multimodal_partial_latent_v2(Dataset):
    """
    Deterministic three-pair dataset (latents), robust to missing modalities.
    
    Seg <-> Color <-> Edge <-> Depth
    
    Pairs:  (seg↔color), (color↔edge), (edge↔depth).

    Parameters
    ----------
    full_overlap_target : int
        Desired overlap shared by ALL THREE pairs (default 45k). Will be clamped
        to the size of the triple-intersection actually available.
    extra_per_pair_target : int
        Desired extra anchors per pair beyond the shared overlap (default 25k),
        applied independently to each pair. If a pair has fewer available IDs,
        it is reduced (not padded with noise).

    __len__()
        By default returns the maximum of the three pair lengths so you can still
        iterate ~70k steps even if the depth pair is smaller; shorter pairs are
        cycled with modulo indexing (deterministic).
    """
    def __init__(
        self,
        resolution,
        root_dir,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        full_overlap_target=45_000,
        extra_per_pair_target=25_000,
        in_channels=3,
        return_filenames=False,
        scale_factor=0.18215,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.return_filenames = return_filenames
        self.scale_factor = scale_factor

        stage = "train"
        self.image_dir        = os.path.join(root_dir, "images",      f"{stage}2017")
        self.sketch_dir       = os.path.join(root_dir, "edges",       f"{stage}2017")
        self.segmentation_dir = os.path.join(root_dir, "annotations", f"{stage}2017")
        self.depthmap_dir     = os.path.join(root_dir, "depthmaps",   f"{stage}2017")

        # Anchor by the color list (sorted, deterministic)
        self.image_paths = _list_tensor_files_recursively(self.image_dir)
        N = len(self.image_paths)
        if N == 0:
            raise ValueError("No color latent .pt files found.")

        # Derived paths (may not exist for some IDs)
        self.sketch_paths = [os.path.join(self.sketch_dir,      Path(p).stem + ".pt")
                             for p in self.image_paths]
        self.seg_paths    = [os.path.join(self.segmentation_dir, Path(p).stem + ".pt")
                             for p in self.image_paths]
        self.depth_paths  = [os.path.join(self.depthmap_dir,    Path(p).stem + "-dpt_beit_large_512.pt")
                             for p in self.image_paths]

        # Availability masks (deterministic: scan in sorted color order)
        ids = np.arange(N, dtype=np.int64)
        ids_ce = np.array([i for i in ids], dtype=np.int64)
        ids_cg = np.array([i for i in ids],    dtype=np.int64)
        ids_ed = np.array([i for i in ids],  dtype=np.int64)

        # Triple-intersection for shared overlap
        ids_triple = np.intersect1d(np.intersect1d(ids_ce, ids_cg), ids_ed, assume_unique=False)
        self.full_overlap = min(int(full_overlap_target), int(len(ids_triple)))
        overlap_ids = ids_triple[:self.full_overlap]  # deterministic prefix



        # Extras per pair: from their available sets excluding overlap AND
        # excluding extras already taken by previous pairs (to make them disjoint).
        def pick_extras(available_ids: np.ndarray, forbidden: np.ndarray, k_target: int) -> np.ndarray:
            rem = np.setdiff1d(available_ids, forbidden, assume_unique=False)
            k = min(int(k_target), int(len(rem)))
            return rem[:k]  # deterministic prefix
        
        used = overlap_ids  # start with the shared overlap as forbidden
        extra_ce = pick_extras(ids_ce, used, extra_per_pair_target)
        used = np.union1d(used, extra_ce)
        
        extra_cg = pick_extras(ids_cg, used, extra_per_pair_target)
        used = np.union1d(used, extra_cg)

        extra_ed = pick_extras(ids_ed, used, extra_per_pair_target)  # reduced if depth is scarce

        # Final per-pair anchor arrays (possibly different lengths!)
        self.anchor_ce = np.concatenate([overlap_ids, extra_ce], axis=0)
        self.anchor_cg = np.concatenate([overlap_ids, extra_cg], axis=0)
        self.anchor_ed = np.concatenate([overlap_ids, extra_ed], axis=0)

        # Keep lengths around
        self.len_ce = len(self.anchor_ce)
        self.len_cg = len(self.anchor_cg)
        self.len_ed = len(self.anchor_ed)

        if self.len_ce == 0 or self.len_cg == 0 or self.len_ed == 0:
            raise ValueError(
                f"Empty pair set(s): CE={self.len_ce}, CG={self.len_cg}, ED={self.len_ed}. "
                "Check your data availability."
            )

        # Dataset length = max so we can still use larger pairs fully.
        self.length = max(self.len_ce, self.len_cg, self.len_ed)

    def __len__(self):
        return self.length

    # ----- latent helpers -----
    def _load_pt(self, p):
        with bf.BlobFile(p, "rb") as f:
            return torch.load(f, map_location="cpu")

    def _sample_moments(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        if z.dim() == 4:
            z.squeeze_(0)
        return z

    def __getitem__(self, idx):
        
        if idx < self.full_overlap:
            # First part of the dataset is the full-overlap set
            idx1 = idx2 = idx3 = idx
        else:        
            # Map idx to each pair deterministically with modulo (cycles shorter pairs)
            idx1 = random.randint(0, self.len_ce - 1)
            idx2 = random.randint(0, self.len_cg - 1)
            idx3 = random.randint(0, self.len_ed - 1)

        i_ce = self.anchor_ce[idx1]
        i_cg = self.anchor_cg[idx2]
        i_ed = self.anchor_ed[idx3]

        # --- color↔edge ---
        z_c_e = self._sample_moments(self._load_pt(self.image_paths[i_ce]))
        z_e   = self._sample_moments(self._load_pt(self.sketch_paths[i_ce]))

        # --- color↔seg ---
        z_c_g = self._sample_moments(self._load_pt(self.image_paths[i_cg]))
        z_g   = self._sample_moments(self._load_pt(self.seg_paths[i_cg]))

        # --- edge↔depth ---
        z_e_d = self._sample_moments(self._load_pt(self.sketch_paths[i_ed]))
        z_d   = self._sample_moments(self._load_pt(self.depth_paths[i_ed]))

        out = torch.cat([z_g, z_c_g, z_c_e, z_e, z_e_d, z_d], dim=0).numpy().astype(np.float32)
    
        out_dict = {
            "color_class": np.array(0, dtype=np.int64),
            "edge_class":  np.array(1, dtype=np.int64),
            "gray_class":  np.array(2, dtype=np.int64),
            "depth_class": np.array(3, dtype=np.int64),
        }
        return out, out_dict