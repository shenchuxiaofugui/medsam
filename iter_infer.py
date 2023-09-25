#%%
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    EnsureTyped,
    ScaleIntensityRanged,
    RepeatChanneld,
    ResizeWithPadOrCropd,
)

from utils.Transform import get_3d_single_box, slice_array, Loadh5, unify_spacing
from monai.data.dataset import CacheDataset
from monai.data.dataloader import DataLoader
import os
join = os.path.join
import torch
from tqdm import tqdm
import h5py
import torch.nn.functional as F
from segment_anything import sam_model_registry
import numpy as np
import nibabel as nib
import json
from utils.utils import crop2
import time
from utils.utils import get_slice
from monai.transforms.spatial.array import ResizeWithPadOrCrop
import argparse

parser = argparse.ArgumentParser(description="iter info 3d")
parser.add_argument("--min_iter", type=int, default=8)
parser.add_argument("--max_iter", type=int, default=10)
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--root", type=str, default="/data3/home/lishengyong/data/ssc_3d/slices")
parser.add_argument("--task_name", type=str, default="")
args = parser.parse_args()
split = args.split
root = args.root

device = f"cuda:0"
model_path = "0818_98"
checkpoint = f"./saved/model/{model_path}/model_best.pth"
# 注册SAM
sam_model = sam_model_registry["vit_b"]().to(device)
sam_model.eval()
info = torch.load(checkpoint)
# sam_model.load_state_dict(info)
my_dic_keys = list(info['state_dict'].keys())
for key in my_dic_keys:
    info['state_dict'][key.replace("module.", "")] = info['state_dict'].pop(key)
sam_model.load_state_dict(info['state_dict'])
sam_model = torch.compile(sam_model)

with open(join(root, "jsons", f"{args.task_name}.json"), "r") as json_file:
    cases = json.load(json_file)    
data_size = len(cases) // 6
if split != 6:
    split_case = cases[split * data_size:(split+1) * data_size]
else:
    split_case = cases[split * data_size:]


def main(iter):
    #设置路径
    axis = iter % 3
    pred_path = f"{root}/infer_3d_iter/axis_1/{model_path}/iter{iter-1}/infer_croped"
    if not os.path.exists(pred_path) and axis == 2:
        pred_path = f"{root}/infer_3d_iter/axis_1/{model_path}/multi_box"
    assert os.path.exists(pred_path), pred_path
    saved_embadding = f"{root}/infer_3d_iter/axis_1/0818_98/iter{iter-3}/embadding"
    clip_min, clip_max = -750, 200
    need_crop = True


    # 这些设置不太用动
    iter_store_path = f"{root}/infer_3d_iter/axis_1/{model_path}/iter{iter}"
    data_path = f"{root}/3D_images"
    store_embadding_path = join(iter_store_path, "embadding")
    store_mask_path = join(iter_store_path, "infer")
    store_croped_path = store_mask_path+"_croped"
    single_box_path = join(iter_store_path, "single_bbox")
    ori_sizes = [(128, 128), (64, 128), (64, 128)]


    #创建目录
    os.makedirs(store_mask_path, exist_ok=True)
    os.makedirs(store_croped_path, exist_ok=True)


    #%% 建立数据列表
    data = []
    for pred in split_case:
        d = {"3D_image": join(data_path, pred+".nii.gz"), "3D_mask": join(pred_path, pred+".nii.gz")}
        data.append(d)

    if axis == 0:
        trans = Compose(
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
        trans = Compose(
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

    ds = CacheDataset(
        data=data,
        transform=trans,
        cache_rate=0,
        num_workers=8,
    )

    test_dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    # 开始编码
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(test_dataloader)):
            imgs_3d, bboxes_3d, masks_3d = (
                            batch_data["3D_image"].to(device), #.to(device)
                            batch_data["bbox"],
                            batch_data["3D_mask"]
                        )
            name = os.path.basename(masks_3d[0].meta["filename_or_obj"])[:-7]
            # nib.save(nib.Nifti1Image(masks_3d[0].numpy(), None), join(trans_mask_path, name+".nii.gz"))
            os.makedirs(join(store_embadding_path, name), exist_ok=True)
            for slice_index in range(len(bboxes_3d)):
                if torch.max(bboxes_3d[slice_index]) == 0:
                    continue          
                bboxes_2d = bboxes_3d[slice_index][0]
                if os.path.exists(join(saved_embadding, name, f"{slice_index}_0.h5py")):
                    with h5py.File(join(saved_embadding, name, f"{slice_index}_0.h5py"), "r") as file:
                        image_embedding = file["img_embedding"][:]
                else:
                    img_2d = slice_array(imgs_3d, axis+2, slice_index)
                    imgs = F.interpolate(
                        img_2d,
                        size=(1024, 1024),
                        mode="bilinear",
                        align_corners=False,
                        )
                    image_embedding = sam_model.image_encoder(imgs)
                    image_embedding = image_embedding.cpu().numpy()
                for number, bbox in enumerate(bboxes_2d):
                    with h5py.File(join(store_embadding_path, name, f"{slice_index}_{number}.h5py"), "w") as file:
                        file["img_embedding"] = image_embedding
                        file["bbox"] = bbox.numpy()
                        


    #%% 解码
    ts_paths = []
    cases = [join(store_embadding_path, i) for i in split_case]
    for case in cases:
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
            bbox_torch = bbox * 8
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
                w, h = singlebbox[3] - singlebbox[1], singlebbox[2] - singlebbox[0]
                w, h = w//10, h//10
                w, h = max(w, 1), max(h, 1)
                bbox_arr[singlebbox[1]:singlebbox[3], singlebbox[0]:singlebbox[2]] = 1
                img_arr = img_arr * 2 + bbox_arr
                new_img_arr = np.zeros_like(img_arr)
                new_img_arr[singlebbox[1]-w:singlebbox[3]+w, singlebbox[0]-h:singlebbox[2]+h] = \
                img_arr[singlebbox[1]-w:singlebbox[3]+w, singlebbox[0]-h:singlebbox[2]+h]
                img_arr = new_img_arr
                img_arr = np.clip(img_arr, 0, 2)
                seg_img = nib.Nifti1Image(img_arr, affine=None)
                parent = os.path.basename(batch_data['parnet'][index])
                os.makedirs(join(single_box_path, parent), exist_ok=True)
                file_store = join(single_box_path, parent, 
                f"{batch_data['name'][index]}.nii.gz")
                nib.save(seg_img, file_store)
                

    # %% 合成   
                
    with open("{root}/ssc_spacing.json", "r") as json_file:
        spacings = json.load(json_file)

    for case_name in split_case:
        case = Path(join(single_box_path, case_name))
        case_arr = []
        name = case.name # .split("_")[0][1:-1]
        z_size = int(64 * spacings[name][2] / spacings[name][0])
        mask_crop = ResizeWithPadOrCrop((z_size, 128))
        cengshu = [64, 128, 128]
        for i in range(cengshu[axis]):
            slice_arr = get_slice(case, i, ori_sizes[axis], mask_crop, axis)
            assert slice_arr.shape == ori_sizes[axis], slice_arr.shape
            case_arr.append(slice_arr)
        mask_arr = np.stack(case_arr, axis=axis)
        pred = nib.Nifti1Image(mask_arr, affine=None)
        nib.save(pred, join(store_mask_path, case.name+".nii.gz"))
        if need_crop:
            img_arr = nib.load(join(data_path, case.name+".nii.gz")).get_fdata()
            mask_arr = mask_arr.astype(int)
            mask_arr[mask_arr==1] = 0
            mask_arr[mask_arr==2] = 1
            crop2_mask_arr = crop2(mask_arr, img_arr, clip_min, clip_max)
            pred = nib.Nifti1Image(crop2_mask_arr, affine=None)
            nib.save(pred, join(store_croped_path, case.name+".nii.gz"))
                
if __name__ == "__main__":
    for i in range(args.min_iter, args.max_iter):
        main(i)