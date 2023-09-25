import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import torch
from skimage import transform
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

from model.med_sam import prompter_network
def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    return np.array([x_min, y_min, x_max, y_max])

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))  

model_type = 'vit_b'
checkpoint = '/data3/home/lishengyong/code/MedSAM/work_dir/ssc_bbox_12/sam_model_val_best.pth'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_prompt = prompter_network(sam_model)

def test_2D(image_data, mask):
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
    image_data_pre[image_data==0] = 0
    img_show_data = image_data_pre
    image_data_pre = transform.resize(
            image_data_pre,
            (256, 256),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )
    image_data_pre = np.repeat(image_data_pre[:,:,None], 3, axis=-1)
    image_data_pre = np.uint8(image_data_pre)
    H, W, _ = image_data_pre.shape
    
    bbox_raw = get_bbox_from_mask(mask)
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    resize_img = sam_transform.apply_image(image_data_pre)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1))
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
    with torch.no_grad():
        # pre-compute the image embedding
        ts_img_embedding = sam_model.image_encoder(input_image)
        # convert box to 1024x1024 grid
        bbox = sam_transform.apply_boxes(bbox_raw, (H, W))
        #print(f'{bbox_raw=} -> {bbox=}')
        box_torch = torch.as_tensor(bbox, dtype=torch.float)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 4) -> (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model.mask_decoder(
            image_embeddings=ts_img_embedding, # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg, img_show_data


if __name__ == "__main__":
    for case in Path("/data3/home/lishengyong/data/ssc/new_slices/test/3D_masks").iterdir():
        mask = sitk.ReadImage(case)
        mask_array = sitk.GetArrayFromImage(mask)
        roi_index = [i for i, e in enumerate(np.sum(mask_array, axis=(0, 1)).tolist()) if e != 0]
        length = len(roi_index)
        if length < 16:
            continue
        img = sitk.ReadImage(str(case).replace("masks", "images"))
        img_array = sitk.GetArrayFromImage(img)
        roi_index = roi_index[int(length/2)-8:int(length/2)+8]
        store_path = str(case).replace("masks", "figure")[:-7]
        os.makedirs(store_path, exist_ok=True)
        for i, index in enumerate(roi_index):
            mask_slice = mask_array[..., index]
            mask_data = transform.resize(
                    mask_slice,
                    (256, 256),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                    )
            
            img_data = img_array[..., index]
            med_seg, img_show = test_2D(img_data, mask_data)
            med_seg = transform.resize(
                    med_seg,
                    (128, 128),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                    )
            j = i % 2
            plt.subplot(2, 2, 2*j+1)
            plt.imshow(img_show)
            plt.contour(med_seg, linewidths = 0.2)
            plt.text(0.5, 0.5, f'pred {index}', fontsize=15, horizontalalignment='left', verticalalignment='top', color='yellow')
            plt.axis('off')
            plt.subplot(2, 2, 2*j+2)
            plt.imshow(img_show)
            plt.contour(mask_slice, linewidths = 0.2)
            plt.text(0.5, 0.5, f'mask {index}', fontsize=15, horizontalalignment='left', verticalalignment='top', color='yellow')
            plt.axis('off')
            if (i+1) % 2 == 0:
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0.1)
                plt.savefig(store_path + f"/{(i+1) // 2}.png",  bbox_inches='tight', dpi = 600)
                plt.clf()
        print(case.name)
            
                
        
    
