#%%
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
from monai.data import DataLoader
from segment_anything import sam_model_registry
from tqdm import tqdm
import os
join=os.path.join
import torch.nn.functional as F
from pathlib import Path
from utils.Transform import get_3d_single_box, slice_array, unify_spacing
import nibabel as nib
import h5py
from utils.Transform import Loadh5
from monai.transforms.spatial.array import ResizeWithPadOrCrop
import matplotlib.pyplot as plt
from utils.utils import get_slice
import argparse

parser = argparse.ArgumentParser(description="iter info 3d")
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--root", type=str, default="/data3/home/lishengyong/data/ssc_3d/slices")
parser.add_argument("--task_name", type=str, default="")
parser.add_argument("--axis", type=int, default=0)
args = parser.parse_args()
split = args.split
slice_store_path = args.root
axis = 0

model_path = "0818_98"
checkpoint = f'/data3/home/lishengyong/code/MedSAM/saved/model/{model_path}/model_best.pth'
device = "cuda:0"
print("拆分：", split, "第几轴：", axis)


#不太用改
axis_path = join(slice_store_path, "infer_3d_iter", f"axis_{axis}")
infer_path = join(axis_path, model_path)
embadding_path = join(axis_path, "embadding")
single_box_path = join(infer_path, "single_box")
multi_box_path = join(infer_path, "multi_box")
ori_sizes = [(128, 128), (64, 128), (64, 128)]

# 创建文件夹
os.makedirs(infer_path, exist_ok=True)
os.makedirs(multi_box_path, exist_ok=True)


#注册SAM
sam_model = sam_model_registry["vit_b"]().to(device)
sam_model.eval()
info = torch.load(checkpoint)
my_dic_keys = list(info['state_dict'].keys())
for key in my_dic_keys:
    info['state_dict'][key.replace("module.", "")] = info['state_dict'].pop(key)
sam_model.load_state_dict(info['state_dict'])
sam_model = torch.compile(sam_model)


# 本次需要infer的数据
with open(join(slice_store_path, "jsons", f"{args.task_name}.json"), "r") as json_file:
    cases = json.load(json_file)
data_size = len(cases) // 5
if split == -1:
    if split != 4:
        split_case = cases[split * data_size:(split+1) * data_size]
    else:
        split_case = cases[split * data_size:]
else:
    split_case = cases
print(len(split_case))


    
#%% 编码
if axis == 0:
    encoder_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["3D_image", "3D_mask"]), 
            EnsureChannelFirstd(keys=["3D_image", "3D_mask"]),
            EnsureTyped(keys=["3D_image"], dtype=torch.float32),
            EnsureTyped(keys=["3D_mask"], dtype=torch.int8),
            ScaleIntensityRanged("3D_image", -1250, 500, 0, 1, True),
            get_3d_single_box(axis),
            RepeatChanneld("3D_image", 3),
        ])
else:
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
            get_3d_single_box(axis),
            RepeatChanneld("3D_image", 3),
        ])     


split_data = [{"3D_image":join(slice_store_path, "3D_images", i+".nii.gz"),
               "3D_mask":join(slice_store_path, "3D_masks", i+".nii.gz")} for i in split_case]

test_encoder_ds = CacheDataset(
    data=split_data,
    transform=encoder_transform,
    cache_rate=0,
    num_workers=8,
)
 

# bbox数量不一样，只能是1，别改
test_encoder_dataloader = DataLoader(test_encoder_ds, batch_size=1, shuffle=False, num_workers=4)

with torch.no_grad():
    for step, batch_data in enumerate(tqdm(test_encoder_dataloader)):
        imgs_3d, bboxes_3d, masks_3d = (
                        batch_data["3D_image"].to(device), #.to(device)
                        batch_data["bbox"],
                        batch_data["3D_mask"]
                    )
        name = os.path.basename(imgs_3d[0].meta["filename_or_obj"])[:-7]
        if os.path.exists(join(multi_box_path, name+".nii.gz")):
            continue
        os.makedirs(join(embadding_path, name), exist_ok=True)
        for slice_index in range(len(bboxes_3d)):
            if os.path.exists(join(embadding_path, name, f"{slice_index}_0.h5py")):
                continue
            if torch.max(bboxes_3d[slice_index]) == 0:
                continue          
            bboxes_2d = bboxes_3d[slice_index][0]
            assert len(bboxes_2d.shape) == 2, f"wrong bbox shape:{bboxes_2d}"
            img_2d = slice_array(imgs_3d, axis+2, slice_index)
            mask_2d = slice_array(masks_3d, axis+2, slice_index)
            imgs = F.interpolate(
                img_2d,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
                )
            image_embedding = sam_model.image_encoder(imgs)
            for number, bbox in enumerate(bboxes_2d):
                with h5py.File(join(embadding_path, name, f"{slice_index}_{number}.h5py"), "w") as file:
                    file["img_embedding"] = image_embedding.cpu().numpy()
                    file["bbox"] = bbox.numpy()

            

#%% 解码

ts_paths = []
for case in split_case:
    if os.path.exists(join(multi_box_path, case+".nii.gz")):
        continue
    case = join(embadding_path, case)
    for slice in Path(case).iterdir():
        ts_paths.append(slice)
print(len(ts_paths))

test_transform = Compose([
                    Loadh5(),
                    ToTensord(["img_embedding", "bbox"], torch.float32)
                ])
val_dataset = CacheDataset(ts_paths, test_transform, cache_rate=0, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=50, shuffle=False)


with torch.no_grad():
    for step, batch_data in enumerate(tqdm(val_dataloader)):       
        image_embedding, bbox = (
            batch_data["img_embedding"].to(device),
            batch_data["bbox"].to(device),
            )
        bbox_torch = bbox * 8  # bbox是记录128 * 128的
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=bbox_torch,
            masks=None,
        )
        medsam_seg_prob, _= sam_model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
                medsam_seg_prob,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
                )
# convert soft mask to hard mask
        ori_res_masks = torch.sigmoid(ori_res_masks)    
        med_seg = ori_res_masks > 0.5
        for index in range(med_seg.shape[0]):
            img_arr = med_seg[index, 0, ...].cpu().numpy().astype(int)
            bbox_arr = np.zeros((128, 128))
            singlebbox = bbox[index].int()
            bbox_arr[singlebbox[1]:singlebbox[3], singlebbox[0]:singlebbox[2]] = 1
            img_arr = img_arr * 2 + bbox_arr
            w, h = singlebbox[3] - singlebbox[1], singlebbox[2] - singlebbox[0]
            w, h = w//10, h//10
            w, h = max(w, 1), max(h, 1)
            new_img_arr = np.zeros_like(img_arr)
            new_img_arr[singlebbox[1]-w:singlebbox[3]+w, singlebbox[0]-h:singlebbox[2]+h] = img_arr[singlebbox[1]-w:singlebbox[3]+w, singlebbox[0]-h:singlebbox[2]+h]
            img_arr = new_img_arr
            img_arr = np.clip(img_arr, 0, 2)
            seg_img = nib.Nifti1Image(img_arr, affine=None)
            parent = os.path.basename(batch_data['parnet'][index])
            os.makedirs(join(single_box_path, parent), exist_ok=True)
            file_store = join(single_box_path, parent, 
            f"{batch_data['name'][index]}.nii.gz")
            nib.save(seg_img, file_store)
            

# %% 合成
                        
with open(f"{slice_store_path}/ssc_spacing.json", "r") as json_file:
    spacings = json.load(json_file)

for case_name in split_case:
    case = Path(join(single_box_path, case_name))
    case_arr = []
    name = case_name
    z_size = int(64 * spacings[name][2] / spacings[name][0])
    mask_crop = ResizeWithPadOrCrop((z_size, 128))
    cengshu = [64, 128, 128]
    for i in range(cengshu[axis]):
        slice_arr = get_slice(case, i, ori_sizes[axis], mask_crop, axis)
        assert slice_arr.shape == ori_sizes[axis], slice_arr.shape
        case_arr.append(slice_arr)
    mask_arr = np.stack(case_arr, axis=axis)    
    pred = nib.Nifti1Image(mask_arr, affine=None)
    nib.save(pred, join(multi_box_path, case_name+".nii.gz"))
    
                    

# %%
