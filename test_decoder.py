# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from monai.data import DataLoader
from segment_anything import sam_model_registry
from monai.metrics import DiceMetric
from model.med_sam import prompter_network
import torch.nn.functional as F
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    ToTensord,
)
from utils.Transform import Loadh5
from monai.data.dataset import CacheDataset
from pathlib import Path
import nibabel as nib

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


split = "test"
path = "old"
#checkpoint = f'/data3/home/lishengyong/code/MedSAM/work_dir/new/ssc_{path}/sam_model_val_best.pth'
device = "cuda:0"
ts_path = "/data3/home/lishengyong/data/ssc_0802/new_slices/aug/old_img_embadding"
save_pred = True
store_path = f"/data3/home/lishengyong/data/ssc_0802/new_slices/aug/infer/{path}"
batch_size = 70
checkpoint = "./saved/model/0807_1536/model_best.pth"



# set up model for fine-tuning 
info = torch.load(checkpoint)
sam_model = sam_model_registry["vit_b"]().to(device)
my_dic_keys = list(info['state_dict'].keys())
for key in my_dic_keys:
    info['state_dict'][key.replace("module.", "")] = info['state_dict'].pop(key)
sam_model.load_state_dict(info['state_dict'])
sam_prompt = prompter_network(sam_model, True).to(device)

# set transform
test_transform = Compose([
                    Loadh5(),
                    ToTensord("label", torch.long),
                    ToTensord(["img_embedding", "bbox"], torch.float32)
                ])


# test
test_files = [i for i in Path(ts_path).iterdir()]
test_dataset = CacheDataset(test_files, transform=test_transform, num_workers=8, cache_rate=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
dice_metric = DiceMetric(include_background=False, reduction="mean")


# validation
sam_model.eval()
with torch.no_grad():
    for step, batch_data in enumerate(tqdm(test_dataloader)):       
    # load data
        image_embedding, gt2D, bbox = (
            batch_data["img_embedding"].to(device),
            batch_data["label"].to(device), batch_data["bbox"].to(device)
            )
        sparse_embeddings, dense_embeddings = sam_prompt(bbox, None)
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
        dice_metric(y_pred=med_seg, y=gt2D)
# convert soft mask to hard mask
        if save_pred:
            os.makedirs(store_path, exist_ok=True)
            for index in range(med_seg.shape[0]):
                seg_img = nib.Nifti1Image(med_seg[index, 0, ...].cpu().float().numpy(), None)
                nib.save(seg_img, join(store_path, f"{batch_data['name'][index]}.nii.gz"))
    print(path, split, dice_metric.aggregate().item())

            

                
        

