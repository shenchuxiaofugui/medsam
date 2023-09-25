from collections.abc import Callable, Sequence
from multiprocessing.managers import ListProxy
import os
import sys
from monai.data.utils import pickle_hashing
import pandas as pd
import numpy as np
import cv2
from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from monai.data.dataset import CacheDataset
from monai.apps.detection.transforms.dictionary import  RandCropBoxByPosNegLabeld
from monai.transforms import (
    LoadImaged,
    Compose,
    Resized,
    RepeatChanneld,
    ToTensord,
    RandLambdad,
    EnsureChannelFirstd,
)
from monai.transforms.transform import Transform
from monai.data.dataset import apply_transform
import SimpleITK as sitk
import h5py
from pathlib import Path
from random import shuffle
from monai.data.dataloader import DataLoader

class MedSamDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        mask_col: str,
        image_dir: Any = None,
        mask_dir: str = None,
        image_size: Tuple = (256, 256),
    ):
        """
        PyTorch dataset class for loading image,mask and bbox pairs from a dataframe.
        The dataframe will need to have atleast two columns for the image and mask file names. The columns can either have the full or relative
        path of the images or just the file names.
        If only file names are given in the columns, the `image_dir` and `mask_dir` arguments should be specified.

        Args:
            df (pd.DataFrame): the pandas dataframe object
            image_col (str): the name of the column on the dataframe that holds the image file names.
            mask_col (str): the name of the column on the dataframe that holds the mask file names.
            image_dir (Any, optional): Path to the input image directory. Defaults to None.
            mask_dir (str, optional): Path to the mask images directory. Defaults to None.
            image_size (Tuple, optional): image size. Defaults to (256, 256).
        """
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_col = image_col
        self.mask_col = mask_col
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # read dataframe row
        row = self.df.iloc[idx]
        # If the `image_dir` attribute is set, the path will be relative to that directory.
        # Otherwise, the path will be the value of the `row[self.image_col]` attribute.
        image_file = (
            os.path.join(self.image_dir, row[self.image_col])
            if self.image_dir
            else row[self.image_col]
        )
        mask_file = (
            os.path.join(self.mask_dir, row[self.mask_col])
            if self.mask_dir
            else row[self.mask_col]
        )

        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Couldn't find image {image_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Couldn't find image {mask_file}")

        # read image and mask files
        image_data = cv2.imread(image_file)
        # read mask as gray scale
        mask_data = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        return self._preprocess(image_data, mask_data)

    def _preprocess(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Threshold mask to binary
        mask = cv2.threshold(mask, 127.0, 255.0, cv2.THRESH_BINARY)[1]
        # convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # min-max normalize and scale
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        # resize
        image = TF.resize(image, self.image_size, antialias=True)
        mask = TF.resize(mask, self.image_size, antialias=True)

        bbox = self._get_bbox(mask)

        return image, mask, bbox

    def _get_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        _, y_indices, x_indices = torch.where(mask > 0)

        x_min, y_min = (x_indices.min(), y_indices.min())
        x_max, y_max = (x_indices.max(), y_indices.max())

        # add perturbation to bounding box coordinates
        H, W = mask.shape[1:]
        # add perfurbation to the bbox
        assert H == W, f"{W} and {H} are not equal size!!"
        x_min = max(0, x_min - np.random.randint(0, 10))
        x_max = min(W, x_max + np.random.randint(0, 10))
        y_min = max(0, y_min - np.random.randint(0, 10))
        y_max = min(H, y_max + np.random.randint(0, 10))

        return torch.tensor([x_min, y_min, x_max, y_max])



class NpzDataset_mask(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        d = np.load(self.data_root)
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = d["gts"]
        self.img_embeddings = d['img_embeddings']
        self.masks = d["masks"]
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")

    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        mask2D = self.masks[index]
        # add perturbation to bounding box coordinates
        
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor([-1]), torch.tensor(mask2D[None, :,:]).float()
    
    
class NpzDataset_bbox(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        d = np.load(self.data_root)
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = d["gts"]
        self.img_embeddings = d['img_embeddings']
        self.masks = d["masks"]

        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        mask2D = self.masks[index]

        y_indices, x_indices = np.where(mask2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), \
        torch.tensor(bboxes).float(), torch.tensor(mask2D[None, :,:]).float()
        
        

class NpzDataset_bbox_bigger(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        d = np.load(self.data_root)
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = d["gts"]
        self.img_embeddings = d['img_embeddings']
        self.masks = d["masks"]

        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        mask2D = self.masks[index]

        y_indices, x_indices = np.where(mask2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        box_x = x_max - x_min
        box_y = y_max - y_min
        # add perturbation to bounding box coordinates
        H, W = mask2D.shape
        # add perfurbation to the bbox
        x_min = max(0, x_min - 5)
        x_max = min(W, x_max + 5)
        y_min = max(0, y_min - 5)
        y_max = min(H, y_max + 5)
        
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), \
        torch.tensor(bboxes).float(), torch.tensor(mask2D[None, :,:]).float()
        
def box_generater(single_slice):
    y_indices, x_indices = torch.where(single_slice[0] > 0)
    x_min, x_max = torch.min(x_indices), torch.max(x_indices)
    y_min, y_max = torch.min(y_indices), torch.max(y_indices)
    return torch.Tensor([x_min, y_min, x_max, y_max])
        
class load_with_box(CacheDataset):
    def __init__(self, data: Sequence, transform: Sequence[Callable[..., Any]] | Callable[..., Any] | None = None,
                 cache_num: int = sys.maxsize, cache_rate: float = 1, num_workers: int | None = 1, 
                 progress: bool = True, copy_cache: bool = True, as_contiguous: bool = True, hash_as_key: bool = False, 
                 hash_func: Callable[..., bytes] = ..., runtime_cache: bool | str | list | ListProxy = False) -> None:
        self.trans2 =Compose([
            Resized(keys=["image", "mask"], spatial_size=(1024, 1024), mode=("brilinear", "nearest")),
            RepeatChanneld("image", 3),
            ToTensord(keys=["image", "bbox"], dtype=torch.float32),
            ToTensord(keys=["mask"], dtype=torch.long)
        ])
        print("dataset size", len(data))
        super().__init__(data, transform, cache_num, cache_rate, num_workers, progress, copy_cache, as_contiguous, hash_as_key, hash_func, runtime_cache)
    
    def __getitem__(self, index: int | slice | Sequence[int]):
        transd_data =  super().__getitem__(index)
        transd_data["ori_img"] = transd_data["image"]
        transd_data["ori_mask"] = transd_data["mask"]
        assert transd_data["image"].shape == (1, 128, 128), "wrong"
        if torch.sum(transd_data["mask"]) == 0:
            transd_data["bbox"] = torch.tensor([[66, 66, 70, 70]])
        else:
            transd_data["bbox"] = box_generater(transd_data["mask"])
        assert transd_data["bbox"].shape[1] == 4, transd_data["bbox"].shape
        return apply_transform(self.trans2, transd_data)
    
    
class load_without_box(CacheDataset):
    def __init__(self, data: Sequence, transform: Sequence[Callable[..., Any]] | Callable[..., Any] | None = None,
                 cache_num: int = sys.maxsize, cache_rate: float = 1, num_workers: int | None = 1, 
                 progress: bool = True, copy_cache: bool = True, as_contiguous: bool = True, hash_as_key: bool = False, 
                 hash_func: Callable[..., bytes] = ..., runtime_cache: bool | str | list | ListProxy = False) -> None:
        self.trans2 =Compose([
            # Resized(keys=["image", "mask"], spatial_size=(1024, 1024), mode=("trilinear", "nearest")),
            RepeatChanneld("image", 3),
            ToTensord(keys=["image"], dtype=torch.float32),
            ToTensord(keys=["mask"], dtype=torch.long)
        ])
        print("dataset size", len(data))
        super().__init__(data, transform, cache_num, cache_rate, num_workers, progress, copy_cache, as_contiguous, hash_as_key, hash_func, runtime_cache)
    
        
        
class NpzDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        d = np.load(self.data_root)
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = d["gts"]
        self.img_embeddings = d['img_embeddings']
        self.sparse_embeddings=d['sparse_embeddings']
        self.dense_embeddings=d['dense_embeddings']
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        sparse_embed = self.sparse_embeddings[index]
        dense_embed = self.dense_embeddings[index]
        # add perturbation to bounding box coordinates
        
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D).long(), torch.tensor(sparse_embed), torch.tensor(dense_embed)
    
    
class Loadh5(Transform):
    def __init__(self):
        super().__init__()
        
    def __call__(self, path: Any):
        d = dict()
        with h5py.File(path, "r") as file:
            d["name"] = path.name[:-5]
            for key,val in file.items():
                if type(val) == h5py._hl.dataset.Dataset:
                    d[key] = val[:]
        if "label" not in d.keys():
            return d
        if len(d["label"].shape) == 2:
            d["label"] = d["label"][None, :, :]
        return d
    
    
class load_one_by_one(CacheDataset):
    def __init__(self, data_root, num_workers, transform, ratio=1, cache_rate=0.3):
        self.data_root = data_root
        files = [i for i in Path(self.data_root).iterdir()]
        shuffle(files)
        size = int(len(files) * ratio)
        self.files = files[:size]
        print("data size", len(self.files))
            
        super().__init__(data=self.files, transform=transform, cache_rate=cache_rate, num_workers=num_workers)
    
    def __len__(self):
        return len(self.files)
    
    
    
if __name__ == "__main__":
    import json
    from monai.data import DataLoader
    with open("/data3/home/lishengyong/data/ssc/new_slices/dataset.json", "r") as file:
        data = json.load(file)
    a = LoadImaged(["image"])
    test_ds = CacheDataset(
    data=data["test"][:10],
    transform=a,
    cache_rate=0,
    num_workers=5,
    )
    test_DataLoader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=5)
    for i in test_DataLoader:
        for j in range(i["image"].shape[0]):
            print(os.path.basename(i["image"][j].meta["filename_or_obj"])[:-7])
        