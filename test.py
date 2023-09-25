#%% 第一步 初始化
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    ToTensord,
    RepeatChanneld,
    RandAffined,
    CenterSpatialCropd,
)
from monai.data import CacheDataset
import json
from monai.data import DataLoader
from segment_anything import sam_model_registry
from tqdm import tqdm
import h5py
import os
join=os.path.join
import torch.nn.functional as F
from pathlib import Path
from utils.Transform import get_single_box
import nibabel as nib
from model.med_sam import prompter_network
from utils.Transform import Loadh5


model_path = "0818_98"
#checkpoint = "/data3/home/lishengyong/code/MedSAM/work_dir/new/ssc_new_sampler/sam_model_val_best.pth"
checkpoint = f'/data3/home/lishengyong/code/MedSAM/saved/model/{model_path}/model_best.pth'
device = "cuda:0"
slice_store_path = "/data3/home/lishengyong/data/ssc_0802/new821"
#slice_store_path = slice_store_path+"/aug"
split = None
split_index = 1


ori_shape = (128, 128)
embadding_store_path = join(slice_store_path, "embadding1")
infer_path = slice_store_path + f"/infer1/{model_path}"
processed_infer_path = join(slice_store_path, f"infer1/{model_path}_processed")

# 创建文件夹
os.makedirs(embadding_store_path, exist_ok=True)
for i in ["images", "masks", "check"]:
    os.makedirs(join(slice_store_path, i), exist_ok=True)

#注册SAM
sam_model = sam_model_registry["vit_b"]().to(device)
sam_model.eval()
info = torch.load(checkpoint)
# sam_model.load_state_dict(info)
my_dic_keys = list(info['state_dict'].keys())
for key in my_dic_keys:
    info['state_dict'][key.replace("module.", "")] = info['state_dict'].pop(key)
sam_model.load_state_dict(info['state_dict'])
    
    
#%% 编码
    
encoder_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "mask"]), 
        EnsureChannelFirstd(keys=["image", "mask"]),
        EnsureTyped(keys=["image"], dtype=np.float32),
        EnsureTyped(keys=["mask"], dtype=torch.long),
        CenterSpatialCropd(keys=["image", "mask"], roi_size=(128, 128)),
        ScaleIntensityRanged("image", -1250, 500, 0, 1, True),
        get_single_box(),
        RepeatChanneld("image", 3),
        ToTensord(keys=["image"], dtype=torch.float32)
    ])


with open(slice_store_path+"/dataset.json") as f:
    data = json.load(f)

if split is not None:
    one_share = len(data["test"]) // split
    datalist = data["test"][split*split_index:split*(split_index+1)]
else:
    datalist = data["test"]
test_encoder_ds = CacheDataset(
    data=datalist,
    transform=encoder_transform,
    cache_rate=0,
    num_workers=8,
)

# bbox数量不一样，只能是1，别改
test_encoder_dataloader = DataLoader(test_encoder_ds, batch_size=1, shuffle=False, num_workers=4)


#%% 正式编码并且保存

with torch.no_grad():
    for step, batch_data in enumerate(tqdm(test_encoder_dataloader)):
        imgs, bboxes, = (
                        batch_data["image"].to(device),
                        batch_data["bbox"],
                    )
        
        imgs = F.interpolate(
                imgs,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
                )
        image_embeddings = sam_model.image_encoder(imgs)
        name = os.path.basename(imgs[0].meta["filename_or_obj"][:-7])
        for number, single_box in enumerate(bboxes[0]):        
            with h5py.File(embadding_store_path + f"/{name}_{number}.h5py", "w") as f:
                f["img_embedding"] = image_embeddings[0, ...].cpu().numpy()
                assert single_box.shape == (4,), single_box.shape
                f["bbox"]= single_box
                
                       
# %% 解码 settting



sam_prompt = prompter_network(sam_model, True, ori_size=ori_shape).to(device)
test_transform = Compose([
                    Loadh5(),
                    ToTensord(["img_embedding", "bbox"], torch.float32)
                ])
test_files = [i for i in Path(embadding_store_path).iterdir()]
print(len(test_files))
val_dataset = CacheDataset(test_files, test_transform, cache_rate=0, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=150, shuffle=False)
os.makedirs(infer_path, exist_ok=True)
sam_model.eval()
with torch.no_grad():
    for step, batch_data in enumerate(tqdm(val_dataloader)):       
        image_embedding, bbox = (
            batch_data["img_embedding"].to(device),
             batch_data["bbox"].to(device)
            )
        sparse_embeddings, dense_embeddings = sam_prompt(bbox, None)
        medsam_seg_prob, _= sam_model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        medsam_seg_prob = F.interpolate(
                medsam_seg_prob,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
                )
# convert soft mask to hard mask
        ori_res_masks = torch.sigmoid(medsam_seg_prob)    
        med_seg = ori_res_masks > 0.5
        for index in range(med_seg.shape[0]):
            img_arr = med_seg[index, 0, ...].cpu().numpy().astype(float)
            bbox_arr = np.zeros(ori_shape)
            single_bbox= bbox[index].int()
            bbox_arr[single_bbox[1]:single_bbox[3], single_bbox[0]:single_bbox[2]] = 1
            img_arr = img_arr * 2 + bbox_arr
            img_arr = np.clip(img_arr, 0, 2)
            seg_img = nib.Nifti1Image(img_arr, affine=None)
            file_store = join(infer_path,
            f"{batch_data['name'][index]}.nii.gz")
            nib.save(seg_img, file_store)

# %% 合成并测试
from utils.SurfaceDice import compute_dice_coefficient
os.makedirs(processed_infer_path, exist_ok=True)
cal_dice = False
with open(slice_store_path+"/dataset.json") as f:
    data = json.load(f)
cases = [os.path.basename(i["image"])[:-7] for i in data["test"]]
cases.sort()
dices = []
check_out = []
for number, case in enumerate(tqdm(cases)):
    sum_array = np.zeros(ori_shape)
    bbox_index = np.zeros(ori_shape)
    mask_index = np.zeros(ori_shape)
    for j in range(10):
        single_file_path = join(infer_path, f"{case}_{j}.nii.gz")
        if os.path.exists(single_file_path):
            roi = nib.load(single_file_path)
            roi_arr = roi.get_fdata()
            bbox_index = np.logical_or(bbox_index, roi_arr == 1)
            mask_index = np.logical_or(mask_index, roi_arr == 2)
        else:
            break
    if j == 0: 
        print(f"hecheng wrong{case}")
        continue
    sum_array[bbox_index] = 1
    sum_array[mask_index] = 2
    if number < 500:
        check_out.append(sum_array)
    # plt.imshow(sum_array)
    # plt.show()
    sum_array = np.pad(sum_array, ((64, 64), (64, 64)), mode='constant')
    case_roi = nib.Nifti1Image(sum_array, affine=None)
    if cal_dice:
        label = nib.load(join(slice_store_path, "labels", case+".nii.gz"))
        label_array = label.get_fdata().astype(np.int16)
        dices.append(compute_dice_coefficient(sum_array == 2, label_array)) 
    nib.save(case_roi, join(processed_infer_path, case+".nii.gz"))
check_img_arr = np.stack(check_out)
check_img = nib.Nifti1Image(check_img_arr, affine=None)
nib.save(check_img, infer_path+".nii.gz")
if cal_dice:
    print(sum(dices)/len(dices))


# %%
