import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Spacingd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    Resized,
    ToTensord,
    RepeatChanneld,
    RandAffined,
)
from monai.data import CacheDataset
import json
import os
join=os.path.join
from pathlib import Path
from utils.Transform import get_3d_single_box, slice_array, unify_spacing
import nibabel as nib

encoder_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["3D_image", "3D_mask"]), 
        EnsureChannelFirstd(keys=["3D_image", "3D_mask"]),
        EnsureTyped(keys=["3D_image"], dtype=torch.float32),
        EnsureTyped(keys=["3D_mask"], dtype=torch.int8),
        ScaleIntensityRanged("3D_image", -1250, 500, 0, 1, True),
        unify_spacing(64),
        ResizeWithPadOrCropd(["3D_image", "3D_mask"], (128, 128, 128), "constant"),
    ])


with open("/data3/home/lishengyong/data/ssc_3d/slices/dataset_3d.json") as f:
    data = json.load(f)


test_encoder_ds = CacheDataset(
    data=data["test"],
    transform=encoder_transform,
    cache_rate=0,
    num_workers=8,
)

image_path = "/data3/home/lishengyong/data/ssc_3d/slices/infer_3d_iter/trans_images"
mask_path = "/data3/home/lishengyong/data/ssc_3d/slices/infer_3d_iter/trans_masks"
os.makedirs(image_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)
for i in test_encoder_ds:
    name = os.path.basename(i["3D_image"].meta["filename_or_obj"])
    nib.save(nib.Nifti1Image(i["3D_image"][0].numpy(), None), join(image_path, name))
    nib.save(nib.Nifti1Image(i["3D_mask"][0].numpy(), None), join(mask_path, name))
    # print(i["3D_mask"].shape)


