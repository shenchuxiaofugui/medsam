# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
# from torch.utils.data import DataLoader
from monai.data import DataLoader
from segment_anything import sam_model_registry
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityRanged,
    RepeatChanneld,
    DataStatsd,
    Resized,
    ToTensord,
)
from monai.data.dataset import CacheDataset, Dataset
import json
from utils.Transform import get_box
import h5py
import torch.nn.functional as F



# %% set up parser

checkpoint = '/data3/home/lishengyong/code/MedSAM/saved/model/0807_1536/model_best.pth'
device = 'cuda:0'
root = "/data3/home/lishengyong/data/ssc_0802/new_slices/aug"

sam_model = sam_model_registry["vit_b"]().to(device)
sam_model.eval()
info = torch.load(checkpoint)
my_dic_keys = list(info['state_dict'].keys())
for key in my_dic_keys:
    info['state_dict'][key.replace("module.", "")] = info['state_dict'].pop(key)
sam_model.load_state_dict(info['state_dict'])


# makedir
os.makedirs(root + f"/old_img_embadding", exist_ok=True)

# transformer
test_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label", "mask"]),
        EnsureChannelFirstd(keys=["image", "label", "mask"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label", "mask"], dtype=torch.long),
        ScaleIntensityRanged("image", -1250, 500, 0, 255, True),
        get_box(),
        RepeatChanneld("image", 3),
        ToTensord(keys=["image", "bbox"], dtype=torch.float32),
        ])

with open(root+"/dataset.json") as f:
    data = json.load(f)

test_ds = CacheDataset(
    data=data["test"],
    transform=test_transform,
    cache_rate=0,
    num_workers=8,
)
size = test_ds.__len__()
print(size)

test_dataloader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4)

for step, batch_data in enumerate(tqdm(test_dataloader)):
    with torch.no_grad():
        img2D, target, bbox = (
                        batch_data["image"].to(device),
                        batch_data["mask"],
                        batch_data["bbox"]  
                    )
        img_1024 = F.interpolate(
                img2D,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
                )    
        image_embedding = sam_model.image_encoder(img_1024)
        for i in range(img2D.shape[0]):
            name = os.path.basename(target[i].meta["filename_or_obj"][:-7])
            with h5py.File(root + f"/old_img_embadding/{name}.h5py", "w") as f:
                f["img_embedding"] = image_embedding[i].cpu().numpy()
                f["bbox"] = bbox[i]
                f["label"] = batch_data["label"][i]
            
    



    

