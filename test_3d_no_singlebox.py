#%% 第一步 得到预处理并适当增强后的图像

import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Spacingd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    ToTensord,
    RepeatChanneld,
    RandAffined,
)
from monai.data import CacheDataset
import json
from monai.data import DataLoader
from segment_anything import sam_model_registry
from tqdm import tqdm
import SimpleITK as sitk
import os
join=os.path.join
import torch.nn.functional as F
from pathlib import Path
from utils.Transform import get_3d_box, get_single_box, slice_array
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.tensorboard import SummaryWriter


model_path = "0807_1536"
# checkpoint = "/data3/home/lishengyong/code/MedSAM/work_dir/new/ssc_new_sampler/sam_model_val_best.pth"
checkpoint = f'/data3/home/lishengyong/code/MedSAM/saved/model/{model_path}/model_best.pth'
device = "cuda:1"
slice_store_path = "/data3/home/lishengyong/data/ssc_0802/new_slices"
axis = 0
infer_path = join(slice_store_path, "infer_3d", f"axis_{axis}", model_path, "center_box")
zooms = [(8, 8), (8, 32), (8, 32)]
ori_sizes = [(128, 128), (32, 128), (32, 128)]
transpose = [(2,1,0), (2,0,1),(0,2,1)]

# 创建文件夹
os.makedirs(infer_path, exist_ok=True)



#注册SAM
sam_model = sam_model_registry["vit_b"]().to(device)
sam_model.eval()
info = torch.load(checkpoint)
# sam_model.load_state_dict(info)
my_dic_keys = list(info['state_dict'].keys())
for key in my_dic_keys:
    info['state_dict'][key.replace("module.", "")] = info['state_dict'].pop(key)
sam_model.load_state_dict(info['state_dict'])
sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)

    
    
#%% 编码

encoder_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["3D_image", "3D_mask"]), 
        EnsureChannelFirstd(keys=["3D_image", "3D_mask"]),
        EnsureTyped(keys=["3D_image"], dtype=torch.float32),
        EnsureTyped(keys=["3D_mask"], dtype=torch.int8),
        ScaleIntensityRanged("3D_image", -1250, 500, 0, 255, True),
        get_3d_box(axis, mode="center"),
        RepeatChanneld("3D_image", 3),
        ToTensord(keys=["3D_image", "bbox"], dtype=torch.float32)
    ])


with open(slice_store_path+"/dataset_3d.json") as f:
    data = json.load(f)

test_encoder_ds = CacheDataset(
    data=data["test"][:100],
    transform=encoder_transform,
    cache_rate=0,
    num_workers=8,
)

# bbox数量不一样，只能是1，别改
test_encoder_dataloader = DataLoader(test_encoder_ds, batch_size=1, shuffle=False, num_workers=4)


#%% 直接infer
k = 1
my_writer = SummaryWriter(log_dir="./work_dir/demo")
with torch.no_grad():
    for step, batch_data in enumerate(tqdm(test_encoder_dataloader)):
        imgs_3d, bboxes_3d, masks_3d = (
                        batch_data["3D_image"].to(device),
                        batch_data["bbox"].to(device),
                        batch_data["3D_mask"],
                    )
        name = os.path.basename(imgs_3d[0].meta["filename_or_obj"])
        segs = []
        assert bboxes_3d.shape[1] == 32
        for slice_index in range(bboxes_3d.shape[1]):
            if slice_index == 16:
                continue
            seg_2d = torch.zeros(ori_sizes[axis]).to(device)
            if torch.max(bboxes_3d[:,slice_index,:]) == 0:
                segs.append(seg_2d)
                continue
            img_2d = slice_array(imgs_3d, axis+2, slice_index)
            assert img_2d.shape[-2:] == ori_sizes[axis], img_2d.shape
            bboxes_2d = bboxes_3d[:,slice_index,:]
            assert bboxes_2d.shape == (1,4), f"wrong bbox shape:{bboxes_2d}"
            imgs = F.interpolate(
                    img_2d,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                    )
            name = os.path.basename(imgs_3d[0].meta["filename_or_obj"][:-7])
            box_torch = torch.empty((1,4)).to(device)
            for i in range(4):
                box_torch[0][i] = bboxes_2d[0][i] * zooms[axis][i % 2]
            my_writer.add_image_with_boxes("big_box4", imgs[0,0,...]/255, box_torch, k, dataformats="HW")
            k += 1
    #         image_embedding = sam_model.image_encoder(imgs)
    #         sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
    #             points=None,
    #             boxes=box_torch,
    #             masks=None,
    #         )
    #         medsam_seg_prob, _= sam_model.mask_decoder(
    #             image_embeddings=image_embedding, # (B, 256, 64, 64)
    #             image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
    #             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
    #             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
    #             multimask_output=False,
    #         )
    #         ori_res_masks = F.interpolate(
    #                 medsam_seg_prob,
    #                 size=ori_sizes[axis],
    #                 mode="bilinear",
    #                 align_corners=False,
    #                 )
    # # convert soft mask to hard mask
    #         ori_res_masks = torch.sigmoid(ori_res_masks)    
    #         med_seg = ori_res_masks > 0.5 
    #         seg_2d = torch.squeeze(med_seg.int())
    #         segs.append(seg_2d)
    #     seg_3d = torch.stack(segs)
    #     seg_array = seg_3d.cpu().numpy()
    #     seg_array = np.transpose(seg_array, transpose[axis])
    #     assert seg_array.shape == (128, 128, 32), seg_array.shape
    #     seg_roi = sitk.GetImageFromArray(seg_array)
    #     sitk.WriteImage(seg_roi, join(infer_path, name+".nii.gz"))
            

