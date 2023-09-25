import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandAffined,
)
from monai.data import CacheDataset
import json
from tqdm import tqdm
import os
join=os.path.join
from pathlib import Path
import nibabel as nib


device = "cuda:0"
slice_store_path = "/data3/home/lishengyong/data/ssc_0802/new_slices"
data_path = slice_store_path+"/dataset.json"
slice_store_path = slice_store_path+"/aug"

# 创建文件夹
for i in ["images", "labels", "masks", "check"]:
    os.makedirs(join(slice_store_path, i), exist_ok=True)

# aug的trans
aug_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label", "mask"]), 
        EnsureChannelFirstd(keys=["image", "label", "mask"]),
        EnsureTyped(keys=["image"], dtype=np.float32),
        RandAffined(keys=["image", "label", "mask"], spatial_size=(128, 128),
                    translate_range=(-40, 40),
                    prob = 0.9,
                    mode=("bilinear", "nearest", "nearest"),
                    padding_mode="zeros"),
    ])

# 更改json文件
with open(data_path) as f:
    data = json.load(f)

aug_data = {"test": []}
for i in data["test"]:
    j = dict()
    for key, value in i.items():
        j[key] = value.replace("new_slices", "new_slices/aug")
    aug_data["test"].append(j)

with open(slice_store_path + "/dataset.json", "w") as file:
    json.dump(aug_data, file)
    
test_aug_ds = CacheDataset(
    data=data["test"],
    transform=aug_transform,
    cache_rate=0,
    num_workers=8,
)

k = 0
for single_data in tqdm(test_aug_ds):
    img2D, mask, gt2D = single_data["image"][0], single_data["mask"][0], single_data["label"][0]
    name = os.path.basename(single_data["label"].meta["filename_or_obj"])
    
    single_img = nib.Nifti1Image(img2D.numpy(), None)
    single_roi = nib.Nifti1Image(gt2D.numpy(), None)
    single_mask = nib.Nifti1Image(mask.numpy(), None)
    nib.save(single_img, join(slice_store_path, "images", name))
    nib.save(single_roi, join(slice_store_path, "labels", name))
    nib.save(single_mask, join(slice_store_path, "masks", name))
    k += 1
    if k<30:
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(img2D)
        plt.contour(gt2D)
        plt.subplot(1,2,2)
        plt.imshow(img2D)
        plt.contour(mask)
        plt.savefig(join(slice_store_path, "check", name[:-7]+".png"), dpi=300)