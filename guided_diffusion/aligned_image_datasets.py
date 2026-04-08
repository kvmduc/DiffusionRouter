import math
import os
import random

import blobfile as bf
import numpy as np
from PIL import Image
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset, Subset
import torch
from pathlib import Path


def load_data(
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
            None,  # resolution is not used in this dataset
            face_sketch_dir,
            face_segment_dir,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels
        )
    elif dataset_name == 'coco_multimodal':
        dataset = COCO_MultiModal_Dataset_latent(
            None,
            data_dir,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels,
            return_filenames=return_filenames
        )
    elif dataset_name == 'coco_multimodal_latent_v2':
        # Latent space for faster training
        dataset = COCO_multimodal_partial_latent_v2(
            None,
            data_dir,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=4,
            return_filenames=return_filenames
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader
        

def load_aligned_data(
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
        
        filepaths = [os.path.basename(path) for path in color_paths]
        
        classes = None
        dataset = AlignedImageDataset(
            image_size,
            edge_paths,
            color_paths,
            gray_paths,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels,
            filepaths=filepaths
        )
    elif dataset_name == 'face_sketch_segment' or dataset_name == 'face_sketch_segment_latent':
        
        face_sketch_dir = os.path.join(data_dir, 'face_sketch')
        face_segment_dir = os.path.join(data_dir, 'face_segment')
        
        edge_paths = _list_image_files_recursively(os.path.join(face_sketch_dir, 'sketch'))
        color_paths = _list_image_files_recursively(os.path.join(face_sketch_dir, 'face'))
        gray_paths = _list_image_files_recursively(os.path.join(face_segment_dir, 'segment'))
        
        filepaths = [os.path.basename(path) for path in color_paths]
        
        classes = None
        dataset = AlignedImageDataset(
            image_size,
            edge_paths,
            color_paths,
            gray_paths,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels,
            filepaths=filepaths
        )
    elif dataset_name == 'coco_multimodal' or dataset_name == 'coco_multimodal_latent' or dataset_name == 'coco_multimodal_latent_v2':
        root_dir = data_dir
        stage = 'test'
        
        # Fully path of datasets
        image_dir = os.path.join(root_dir, 'images', f'{stage}2017')
        sketch_dir = os.path.join(root_dir, 'edges', f'{stage}2017')
        segmentation_dir = os.path.join(root_dir, 'annotations', f'{stage}2017')
        depthmap_dir = os.path.join(root_dir, 'depthmaps', f'{stage}2017')

        image_paths = _list_image_files_recursively(image_dir)
        sketch_paths = [os.path.join(sketch_dir, Path(p).stem + '.png') for p in image_paths]
        segmentation_paths = [os.path.join(segmentation_dir, Path(p).stem + '.png') for p in image_paths]
        depthmap_paths = [os.path.join(depthmap_dir, Path(p).stem + '-dpt_beit_large_512.png') for p in image_paths]
        
        filepaths = [os.path.basename(path) for path in image_paths]
        
        classes = None
        dataset = AlignedImageDataset(
            image_size,
            sketch_paths,
            image_paths,
            segmentation_paths,
            depth_paths=depthmap_paths,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            in_channels=in_channels,
            filepaths=filepaths,
            dataset_name=dataset_name
        )
    else: 
        raise ValueError(f"unknown dataset: {dataset_name}")
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def list_image_files(data_dir):
    """List images files in the directory (not recursively)."""
    files = sorted(bf.listdir(data_dir))
    results = []
    for entry in files:
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
    return results


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def _list_tensor_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["pt", "pth"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_tensor_files_recursively(full_path))
    return results


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
        
        # Train with 2 flow, forward and backward pair
        self.target_data = self.color1_paths + self.color2_paths + self.edge_paths + self.gray_paths
        self.context_data = self.edge_paths + self.gray_paths + self.color1_paths + self.color2_paths
        
        """
        class index:
        0: x_color                      common domain
        1: x_edge
        2: x_grayscale
        """
        self.target_classes = [0] * len(self.color1_paths) + [0] * len(self.color2_paths) + [1] * len(self.edge_paths) + [2] * len(self.gray_paths)
        self.context_classes = [1] * len(self.edge_paths) + [2] * len(self.gray_paths) + [0] * len(self.color1_paths) + [0] * len(self.color2_paths)

    
    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        target_path = self.target_data[idx]
        context_path = self.context_data[idx]
        
        target_class = self.target_classes[idx]
        context_class = self.context_classes[idx]
        
        with bf.BlobFile(target_path, "rb") as f:
            target_img = Image.open(f)
            target_img.load()
            
        with bf.BlobFile(context_path, "rb") as f:
            context_img = Image.open(f)
            context_img.load()
            
        if self.in_channels == 1:
            target_img = target_img.convert("L")
            context_img = context_img.convert("L")
        elif self.in_channels == 3:
            target_img = target_img.convert("RGB")
            context_img = context_img.convert("RGB")
        
        target_img = target_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        target_img = np.array(target_img)
        context_img = context_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        context_img = np.array(context_img)
                    
        if len(target_img.shape) < 3:
            target_img = target_img[:, :, np.newaxis]  # Adds a single channel
        if len(context_img.shape) < 3:
            context_img = context_img[:, :, np.newaxis]  # Adds a single channel
            
        if self.random_flip and random.random() < 0.5:
            target_img = target_img[:, ::-1]
            context_img = context_img[:, ::-1]
            
        target_img = target_img.astype(np.float32) / 127.5 - 1
        context_img = context_img.astype(np.float32) / 127.5 - 1
        
        target_img = np.transpose(target_img, [2, 0, 1])                        # transpose to 3x64x64
        context_img = np.transpose(context_img, [2, 0, 1])                        # transpose to 3x64x64
        
        out_dict = dict()
        out_dict["target_class"] = np.array(target_class, dtype=np.int64)
        out_dict["context_class"] = np.array(context_class, dtype=np.int64)
        
        if self.return_filenames:
            out_dict['target_filenames'] = os.path.basename(target_path)
            out_dict['context_filenames'] = os.path.basename(context_path)
        
        return np.vstack((target_img, context_img), dtype=np.float32), out_dict                         # return 9x64x64 image


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

    def __getitem__(self, idx):
        target_path = self.target_data[idx]
        context_path = self.context_data[idx]
        
        target_class = self.target_classes[idx]
        context_class = self.context_classes[idx]
        
        with bf.BlobFile(target_path, "rb") as f:
            target_img = Image.open(f)
            target_img.load()
            
        with bf.BlobFile(context_path, "rb") as f:
            context_img = Image.open(f)
            context_img.load()
            
        if self.in_channels == 1:
            target_img = target_img.convert("L")
            context_img = context_img.convert("L")
        elif self.in_channels == 3:
            target_img = target_img.convert("RGB")
            context_img = context_img.convert("RGB")
        
        target_img = target_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        target_img = np.array(target_img)
        context_img = context_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        context_img = np.array(context_img)
        
        if len(target_img.shape) < 3:
            target_img = target_img[:, :, np.newaxis]  # Adds a single channel
        if len(context_img.shape) < 3:
            context_img = context_img[:, :, np.newaxis]  # Adds a single channel
            
        if self.random_flip and random.random() < 0.5:
            target_img = target_img[:, ::-1]
            context_img = context_img[:, ::-1]
            
        target_img = target_img.astype(np.float32) / 127.5 - 1
        context_img = context_img.astype(np.float32) / 127.5 - 1
        
        target_img = np.transpose(target_img, [2, 0, 1])                        # transpose to 3x64x64
        context_img = np.transpose(context_img, [2, 0, 1])                        # transpose to 3x64x64
        
        out_dict = dict()
        out_dict["target_class"] = np.array(target_class, dtype=np.int64)
        out_dict["context_class"] = np.array(context_class, dtype=np.int64)
        
        if self.return_filenames:
            out_dict['target_filenames'] = os.path.basename(target_path)
            out_dict['context_filenames'] = os.path.basename(context_path)
        
        return np.vstack((target_img, context_img), dtype=np.float32), out_dict
    

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
        target_path = self.target_data[idx]
        context_path = self.context_data[idx]
        
        target_class = self.target_classes[idx]
        context_class = self.context_classes[idx]
        
        with bf.BlobFile(target_path, "rb") as f:
            target_tensor = torch.load(f, map_location='cpu')           # (8, 32, 32)
            
        with bf.BlobFile(context_path, "rb") as f:
            context_tensor = torch.load(f, map_location='cpu')          # (8, 32, 32)

        target_tensor = self._sample_moments(target_tensor).numpy()       # (4, 32, 32)
        context_tensor = self._sample_moments(context_tensor).numpy()     # (4, 32, 32)

        out_dict = dict()
        out_dict["target_class"] = np.array(target_class, dtype=np.int64)
        out_dict["context_class"] = np.array(context_class, dtype=np.int64)
        
        if self.return_filenames:
            out_dict['target_filenames'] = os.path.basename(target_path)
            out_dict['context_filenames'] = os.path.basename(context_path)

        return np.vstack((target_tensor, context_tensor), dtype=np.float32), out_dict


class COCO_MultiModal_Dataset(Dataset):
    def __init__(self,
            resolution,
            root_dir,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            in_channels=3,
            filepaths=None,
            return_filenames=False):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.filepaths = filepaths
        self.return_filenames = return_filenames
        self.root_dir = root_dir
        stage = 'train'
        
        # Fully path of datasets
        self.image_dir = os.path.join(self.root_dir, 'images', f'{stage}2017')
        self.sketch_dir = os.path.join(self.root_dir, 'edges', f'{stage}2017')
        self.segmentation_dir = os.path.join(self.root_dir, 'annotations', f'{stage}2017')
        self.depthmap_dir = os.path.join(self.root_dir, 'depthmaps', f'{stage}2017')


        self.image_paths = _list_image_files_recursively(self.image_dir)
        self.sketch_paths = [os.path.join(self.sketch_dir, Path(p).stem + '.png') for p in self.image_paths]
        self.segmentation_paths = [os.path.join(self.segmentation_dir, Path(p).stem + '.png') for p in self.image_paths]
        self.depthmap_paths = [os.path.join(self.depthmap_dir, Path(p).stem + '-dpt_beit_large_512.png') for p in self.image_paths]
        
        # Train with 2 flow, forward and backward pair
        self.target_data = self.image_paths + self.image_paths + self.image_paths + self.sketch_paths + self.segmentation_paths + self.depthmap_paths
        self.context_data = self.sketch_paths + self.segmentation_paths + self.depthmap_paths + self.image_paths + self.image_paths + self.image_paths

        """
        class index:
        0: x_color                      common domain
        1: x_sketch
        2: x_segment
        3: x_depthmap
        """
        self.target_classes = [0] * len(self.image_paths) + [0] * len(self.image_paths) + [0] * len(self.image_paths) + [1] * len(self.sketch_paths) + [2] * len(self.segmentation_paths) + [3] * len(self.depthmap_paths)
        self.context_classes = [1] * len(self.sketch_paths) + [2] * len(self.segmentation_paths) + [3] * len(self.depthmap_paths) + [0] * len(self.image_paths) + [0] * len(self.image_paths) + [0] * len(self.image_paths)

        assert len(self.target_classes) == len(self.context_classes) 
        assert len(self.target_data) == len(self.context_data)


    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        target_path = self.target_data[idx]
        context_path = self.context_data[idx]
        
        target_class = self.target_classes[idx]
        context_class = self.context_classes[idx]
        
        with bf.BlobFile(target_path, "rb") as f:
            target_img = Image.open(f)
            target_img.load()
            
        with bf.BlobFile(context_path, "rb") as f:
            context_img = Image.open(f)
            context_img.load()
            
        if self.in_channels == 1:
            target_img = target_img.convert("L")
            context_img = context_img.convert("L")
        elif self.in_channels == 3:
            target_img = target_img.convert("RGB")
            context_img = context_img.convert("RGB")

        target_img = target_img.resize((self.resolution, self.resolution), Image.NEAREST if target_class == 2 else Image.BILINEAR)
        target_img = np.array(target_img)
        context_img = context_img.resize((self.resolution, self.resolution), Image.NEAREST if target_class == 2 else Image.BILINEAR)
        context_img = np.array(context_img)
        
        if len(target_img.shape) < 3:
            target_img = target_img[:, :, np.newaxis]  # Adds a single channel
        if len(context_img.shape) < 3:
            context_img = context_img[:, :, np.newaxis]  # Adds a single channel
            
        if self.random_flip and random.random() < 0.5:
            target_img = target_img[:, ::-1]
            context_img = context_img[:, ::-1]
            
        target_img = target_img.astype(np.float32) / 127.5 - 1
        context_img = context_img.astype(np.float32) / 127.5 - 1
        
        target_img = np.transpose(target_img, [2, 0, 1])                        # transpose to 3x64x64
        context_img = np.transpose(context_img, [2, 0, 1])                        # transpose to 3x64x64
        
        out_dict = dict()
        out_dict["target_class"] = np.array(target_class, dtype=np.int64)
        out_dict["context_class"] = np.array(context_class, dtype=np.int64)
        
        if self.return_filenames:
            out_dict['target_filenames'] = os.path.basename(target_path)
            out_dict['context_filenames'] = os.path.basename(context_path)
        
        return np.vstack((target_img, context_img), dtype=np.float32), out_dict

    
    
class COCO_MultiModal_Dataset_latent(Dataset):
    def __init__(self,
            resolution,
            root_dir,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            in_channels=3,
            filepaths=None,
            scale_factor=0.18215,
            return_filenames=False):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.filepaths = filepaths
        self.return_filenames = return_filenames
        self.root_dir = root_dir
        self.scale_factor = scale_factor
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

        # Train with 2 flow, forward and backward pair
        self.target_data = self.image_paths + self.image_paths + self.image_paths + self.sketch_paths + self.segmentation_paths + self.depthmap_paths
        self.context_data = self.sketch_paths + self.segmentation_paths + self.depthmap_paths + self.image_paths + self.image_paths + self.image_paths

        """
        class index:
        0: x_color                      common domain
        1: x_sketch
        2: x_segment
        3: x_depthmap
        """
        self.target_classes = [0] * len(self.image_paths) + [0] * len(self.image_paths) + [0] * len(self.image_paths) + [1] * len(self.sketch_paths) + [2] * len(self.segmentation_paths) + [3] * len(self.depthmap_paths)
        self.context_classes = [1] * len(self.sketch_paths) + [2] * len(self.segmentation_paths) + [3] * len(self.depthmap_paths) + [0] * len(self.image_paths) + [0] * len(self.image_paths) + [0] * len(self.image_paths)

        assert len(self.target_classes) == len(self.context_classes) 
        assert len(self.target_data) == len(self.context_data)
        assert len(self.image_paths) == len(self.sketch_paths) == len(self.segmentation_paths) == len(self.depthmap_paths), \
            "Every image must have a sketch, segmentation & depth latent file."


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
        target_path = self.target_data[idx]
        context_path = self.context_data[idx]
        
        target_class = self.target_classes[idx]
        context_class = self.context_classes[idx]
        
        with bf.BlobFile(target_path, "rb") as f:
            target_tensor = torch.load(f, map_location='cpu')           # (8, 32, 32)
            
        with bf.BlobFile(context_path, "rb") as f:
            context_tensor = torch.load(f, map_location='cpu')          # (8, 32, 32)

        target_tensor = self._sample_moments(target_tensor).numpy()       # (4, 32, 32)
        context_tensor = self._sample_moments(context_tensor).numpy()     # (4, 32, 32)

        out_dict = dict()
        out_dict["target_class"] = np.array(target_class, dtype=np.int64)
        out_dict["context_class"] = np.array(context_class, dtype=np.int64)
        
        if self.return_filenames:
            out_dict['target_filenames'] = os.path.basename(target_path)
            out_dict['context_filenames'] = os.path.basename(context_path)

        return np.vstack((target_tensor, context_tensor), dtype=np.float32), out_dict
    
    
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
        direction_mode="random",   # "alternate" or "random"
        pair_mode="random",       # "roundrobin" or "random"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.return_filenames = return_filenames
        self.scale_factor = scale_factor
        
        self.direction_mode = direction_mode
        self.pair_mode = pair_mode
        self.CLASS_IDX = {"color": 0, "edge": 1, "gray": 2, "depth": 3}

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

    def _pick_pair(self, idx):
        # Choose which of the three pairs this sample represents
        if self.pair_mode == "random":
            which = random.randint(0, 2)
        else:  # roundrobin
            which = idx % 3

        if which == 0:  # color↔seg
            i = self.anchor_cg[idx % self.len_cg]
            pair = "cg"
            color_pt = self._load_pt(self.image_paths[i])
            seg_pt   = self._load_pt(self.seg_paths[i])
            z_color  = self._sample_moments(color_pt)
            z_seg    = self._sample_moments(seg_pt)
            a, b = ("color", z_color), ("gray", z_seg)

        elif which == 1:  # color↔edge
            i = self.anchor_ce[idx % self.len_ce]
            pair = "ce"
            color_pt = self._load_pt(self.image_paths[i])
            edge_pt  = self._load_pt(self.sketch_paths[i])
            z_color  = self._sample_moments(color_pt)
            z_edge   = self._sample_moments(edge_pt)
            a, b = ("color", z_color), ("edge", z_edge)

        else:  # which == 2, edge↔depth
            i = self.anchor_ed[idx % self.len_ed]
            pair = "ed"
            edge_pt  = self._load_pt(self.sketch_paths[i])
            depth_pt = self._load_pt(self.depth_paths[i])
            z_edge   = self._sample_moments(edge_pt)
            z_depth  = self._sample_moments(depth_pt)
            a, b = ("edge", z_edge), ("depth", z_depth)

        return pair, a, b, i

    def __getitem__(self, idx):
        pair, a, b, i = self._pick_pair(idx)
        (name_a, z_a), (name_b, z_b) = a, b  # each 4xHxW

        # Choose direction (who is target vs context)
        if self.direction_mode == "random":
            flip = bool(random.getrandbits(1))
        else:  # alternate deterministically
            flip = (idx // 3) % 2 == 1

        if flip:
            target_name, target_z = name_b, z_b
            context_name, context_z = name_a, z_a
        else:
            target_name, target_z = name_a, z_a
            context_name, context_z = name_b, z_b

        # Merge as (target || context) along channels -> (8,H,W)
        pair_z = torch.cat([target_z, context_z], dim=0).to(torch.float32)

        out = pair_z.numpy().astype(np.float32)
        out_dict = {
            "target_class": np.array(self.CLASS_IDX[target_name], dtype=np.int64),
            "context_class": np.array(self.CLASS_IDX[context_name], dtype=np.int64),
        }
        if self.return_filenames:
            stem = Path(self.image_paths[i]).stem
            out_dict["stem"] = stem  # base COCO id
        return out, out_dict
    

class AlignedImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            edge_paths,
            color_paths,
            gray_paths,
            depth_paths=None,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            in_channels=3,
            filepaths=None,
            dataset_name=None
    ):
        super().__init__()
        self.resolution = resolution
        self.edge_paths = edge_paths[shard:][::num_shards]
        self.color_paths = color_paths[shard:][::num_shards]
        self.gray_paths = gray_paths[shard:][::num_shards]
        
        self.depth_paths = depth_paths[shard:][::num_shards] if depth_paths is not None else None

        self.local_classes = None
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.filepaths = filepaths
        self.dataset_name = dataset_name
    
    
    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        edge_path = self.edge_paths[idx]
        color_path = self.color_paths[idx]
        gray_path = self.gray_paths[idx]
        with bf.BlobFile(edge_path, "rb") as f:
            edge_img = Image.open(f)
            edge_img.load()
        with bf.BlobFile(color_path, "rb") as f:
            color_img = Image.open(f)
            color_img.load()
        with bf.BlobFile(gray_path, "rb") as f:
            gray_img = Image.open(f)
            gray_img.load()
            
        if self.in_channels == 3:
            edge_img = edge_img.convert("RGB")
            color_img = color_img.convert("RGB")
            gray_img = gray_img.convert("RGB")
        elif self.in_channels == 1:
            edge_img = edge_img.convert("L")
            color_img = color_img.convert("L")
            gray_img = gray_img.convert("L")
        
        is_segmentation = self.dataset_name == 'coco_multimodal'
        
        edge_img = edge_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        edge_img = np.array(edge_img)
        color_img = color_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        color_img = np.array(color_img)
        gray_img = gray_img.resize((self.resolution, self.resolution), Image.BILINEAR if not is_segmentation else Image.NEAREST)
        gray_img = np.array(gray_img)
            
        if len(edge_img.shape) < 3:
            edge_img = edge_img[:, :, np.newaxis]  # Adds a single channel
        if len(color_img.shape) < 3:
            color_img = color_img[:, :, np.newaxis]  # Adds a single channel
        if len(gray_img.shape) < 3:
            gray_img = gray_img[:, :, np.newaxis]  # Adds a single channel

        if self.random_flip and random.random() < 0.5:
            edge_img = edge_img[:, ::-1]
            color_img = color_img[:, ::-1]
            gray_img = gray_img[:, ::-1]

        edge_img = edge_img.astype(np.float32) / 127.5 - 1
        color_img = color_img.astype(np.float32) / 127.5 - 1
        gray_img = gray_img.astype(np.float32) / 127.5 - 1
        
        edge_img = np.transpose(edge_img, [2, 0, 1])                        # transpose to 3x64x64
        color_img = np.transpose(color_img, [2, 0, 1])                      # transpose to 3x64x64
        gray_img = np.transpose(gray_img, [2, 0, 1])                        # transpose to 3x64x64
        
        out_dict = dict()
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.filepaths is not None:
            out_dict["filepath"] = self.filepaths[idx]
        
        if self.depth_paths is not None:
            depth_path = self.depth_paths[idx]
            with bf.BlobFile(depth_path, "rb") as f:
                depth_img = Image.open(f)
                depth_img.load()
            if self.in_channels == 1:
                depth_img = depth_img.convert("L")
            elif self.in_channels == 3:
                depth_img = depth_img.convert("RGB")
            depth_img = depth_img.resize((self.resolution, self.resolution), Image.BILINEAR)
            depth_img = np.array(depth_img)
            
            if len(depth_img.shape) < 3:
                depth_img = depth_img[:, :, np.newaxis]
            if self.random_flip and random.random() < 0.5:
                depth_img = depth_img[:, ::-1]
            depth_img = depth_img.astype(np.float32) / 127.5 - 1
            depth_img = np.transpose(depth_img, [2, 0, 1])

            return np.vstack((color_img, edge_img, gray_img, depth_img)), out_dict  # return 12x64x64 image

        return np.vstack((color_img, edge_img, gray_img)), out_dict                         # return 9x64x64 image


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def get_image_filenames_for_label(label):
    """
    Returns the validation files for images with the given label. This is a utility
    function for ImageNet translation experiments.
    :param label: an integer in 0-1000
    """
    # First, retrieve the synset word corresponding to the given label
    base_dir = os.getcwd()
    synsets_filepath = os.path.join(base_dir, "evaluations", "synset_words.txt")
    synsets = [line.split()[0] for line in open(synsets_filepath).readlines()]
    synset_word_for_label = synsets[label]

    # Next, build the synset to ID mapping
    synset_mapping_filepath = os.path.join(base_dir, "evaluations", "map_clsloc.txt")
    synset_to_id = dict()
    with open(synset_mapping_filepath) as file:
        for line in file:
            synset, class_id, _ = line.split()
            synset_to_id[synset.strip()] = int(class_id.strip())
    true_label = synset_to_id[synset_word_for_label]

    # Finally, return image files corresponding to the true label
    validation_ground_truth_filepath = os.path.join(base_dir, "evaluations", "ILSVRC2012_validation_ground_truth.txt")
    source_data_labels = [int(line.strip()) for line in open(validation_ground_truth_filepath).readlines()]
    image_indexes = [i + 1 for i in range(len(source_data_labels)) if true_label == source_data_labels[i]]
    output = [f"ILSVRC2012_val_{str(i).zfill(8)}.JPEG" for i in image_indexes]
    return output
