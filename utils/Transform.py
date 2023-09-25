from collections.abc import Sequence
from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.transforms.transform import MapTransform, Transform
from typing import Any, Iterator, Optional, Sized
import h5py
from monai.utils import GridSampleMode, GridSamplePadMode
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
from skimage import measure
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from monai.transforms.spatial.array import Spacing, Resize
# from monai.apps.detection.transforms.box_ops import convert_box_to_mask

def box_jitter(bbox):
    box_x = bbox[2] - bbox[0]
    box_y = bbox[3] - bbox[1]
    rand = np.random.uniform(low=-0.3, high=0.3, size=4)
    bbox[0] = bbox[0] + rand[0]*box_x
    bbox[1] = bbox[1] + rand[1]*box_y
    bbox[2] = bbox[2] + rand[2]*box_x
    bbox[3] = bbox[3] + rand[3]*box_y
    # bbox = bbox + np.random.randint(-3, 4)
    new_bbox = np.clip(bbox, 0, 128)
    return new_bbox    

def box_zoom(bbox):
    box_x = bbox[2] - bbox[0]
    box_y = bbox[3] - bbox[1]
    rand = np.random.uniform(low=-0.2, high=0.2, size=2)
    bbox[0] = bbox[0] - rand[0]*box_x
    bbox[1] = bbox[1] - rand[1]*box_y
    bbox[2] = bbox[2] + rand[0]*box_x
    bbox[3] = bbox[3] + rand[1]*box_y
    # bbox = bbox + np.random.randint(-3, 4)
    new_bbox = np.clip(bbox, 0, 128)
    return new_bbox  


class Loadh5(Transform):
    def __init__(self):
        super().__init__()
        
    def __call__(self, path: Any):
        d = dict()
        with h5py.File(path, "r") as file:
            d["name"] = path.name[:-5]
            d["parnet"] = str(path.parent)
            for key,val in file.items():
                if type(val) == h5py._hl.dataset.Dataset:
                    d[key] = val[:]
        if len(d["img_embedding"].shape) != 3:
            d["img_embedding"] = d["img_embedding"][0]
        if "label" not in d.keys():
            return d
        if len(d["label"].shape) == 2:
            d["label"] = d["label"][None, :, :]
        return d
    
class Loadh5_v2(Transform):
    def __init__(self):
        super().__init__()  
        
    def __call__(self, data: Any):
        keys = [i for i in data.keys()]
        assert len(keys) == 1, "load wrong"
        key = keys[0]
        number = key.split("_")[-1]
        filepath = key[:-(len(number)+1)]
        d = dict()
        with h5py.File(filepath, "r") as f:
            d["img_embedding"] = f["img_embedding"][:]
        d["bbox"] = data[key]
        d["number"] = number
        roi = sitk.ReadImage(filepath.replace("embadding", "rois").replace("h5py", "nii.gz"))
        d["label"] = sitk.GetArrayFromImage(roi) 
        d["name"] = os.path.basename(filepath)          
        return d  
          


def box_generater(single_slice):
    if torch.max(single_slice) == 0:
        return torch.Tensor([62, 62, 67, 67])
    y_indices, x_indices = torch.where(single_slice > 0)
    x_min, x_max = torch.min(x_indices), torch.max(x_indices)
    y_min, y_max = torch.min(y_indices), torch.max(y_indices)
    return torch.Tensor([x_min, y_min, x_max, y_max])
    
    
class get_box(Transform):
    def __init__(self):
        super().__init__()
    
    def __call__(self, data):
        d = dict(data)
        d["bbox"] = box_generater(d["mask"][0])
        return d
    


def mask_shift(org_image, org_mask):
    labeled_org_mask = measure.label(org_mask, connectivity=2)

    # Padded image and padded connected components of mask
    padded_org_image = np.pad(org_image, ((128, 128), (128, 128)))
    padded_labeled_org_mask = np.pad(labeled_org_mask, ((128, 128), (128, 128)))

    # Connected component props on padded original mask
    padded_ccs = measure.regionprops(padded_labeled_org_mask)

    # For output result
    new_imgs, centers, bboxes, masks = [], [], [], []
    for i, cc in enumerate(padded_ccs):
        mask = padded_labeled_org_mask == i + 1  # start from 1
        bbox = cc.bbox
        center_x, center_y =  (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2
        # Crop image and mask for inference
        cropped_image = padded_org_image[center_x - 64: center_x + 64, center_y - 64: center_y + 64]
        cropped_mask = mask[center_x - 64: center_x + 64, center_y - 64: center_y + 64]
        

        # Coordinats transform between original image and cropped image
        # shift_x, shift_y = center_x - 128 - 64, center_y - 128 - 64

        # Current bbox on cropped image (128, 128)
        bbox = measure.regionprops(cropped_mask * 1)[0].bbox
        
        new_imgs.append(cropped_image)
        centers.append((center_x - 128, center_y - 128))
        bboxes.append(bbox)
        masks.append(cropped_mask)
    
    return np.stack(new_imgs), centers, bboxes, np.stack(masks)


def shift_mask(cropped_mask, shifts):
    coords = np.transpose(np.where(cropped_mask))  # Nx3
    coords += shifts  # coordinates transform original image
    # do clipping, in case of out of index error
    legal_index = np.logical_and((coords < 128).all(1), (coords > 0).all(1))  # 
    coords = coords[legal_index]
    return tuple(coords.T)
    # Mapping


class get_center_box(Transform):
    # 把mask和box移到中间
    def __init__(self):
        super().__init__()
    
    def __call__(self, data):
        d = dict(data)
        new_imgs, centers, bboxes, _ = mask_shift(d["image"][0], d["mask"][0])
        assert len(new_imgs.shape) == 3, "wrong shape"
        d["shifted_imgs"] = new_imgs
        d["centers"] = centers
        d["bbox"] = bboxes
        return d


def center_bbox_generater(mask):
    labeled_mask = measure.label(mask)
    ccs = measure.regionprops(labeled_mask)
    mask_center_x = mask.shape[0] // 2
    mask_center_y = mask.shape[1] // 2
    center = 10000
    center_bbox = None
    for i in ccs:
        center_x, center_y = i.centroid
        if (center_x-mask_center_x) ** 2 + (center_y-mask_center_y) ** 2 < center:
            center = center_x ** 2 + center_y ** 2
            x0, y0, x1, y1 = i.bbox
            center_bbox = (y0, x0, y1, x1)
    return torch.tensor(center_bbox)


def single_box_generater(mask):
    labeled_mask = measure.label(mask)
    ccs = measure.regionprops(labeled_mask)
    bboxes= []
    for i in ccs:
        x0, y0, x1, y1 = i.bbox
        bboxes.append((y0, x0, y1, x1))
    return torch.tensor(bboxes)
    
    
class get_single_box(Transform):
    # 直接获取box
    def __init__(self):
        super().__init__()
    
    def __call__(self, data):
        d = dict(data)
        bboxes = single_box_generater(d["mask"][0].numpy())
        d["bbox"] = bboxes
        return d


def slice_array(arr, dim, index):
    if dim < 0 or dim >= len(arr.shape):
        raise ValueError("Invalid dimension specified")

    slices = [slice(None)] * len(arr.shape)
    slices[dim] = index
    return arr[tuple(slices)]    
 
    
class get_3d_box(Transform):
    # 直接获取box
    def __init__(self, axis, aug=None, mode="all"):
        super().__init__()
        self.axis = axis
        self.aug = aug
        self.mode = mode
    
    def __call__(self, data):
        d = dict(data)
        bboxes = []
        img = d["3D_mask"][0]
        exist = []
        for i in range(img.shape[self.axis]):
            slice_img = slice_array(img, self.axis, i)
            if torch.max(slice_img) == 0:
                bbox_2d = torch.zeros((4))
            else:
                if self.mode == "all":
                    bbox_2d = box_generater(slice_img)
                elif self.mode == "center":
                    bbox_2d = center_bbox_generater(slice_img)
                exist.append(i)
            bboxes.append(bbox_2d)
        if self.aug is not None:
            min_i = exist[0]
            max_i = exist[-1]
            for i in range(self.aug):
                if min_i - i - 1 < 0 or max_i + i + 1 > 31:
                    break
                bboxes[min_i - i - 1] = bboxes[min_i]
                bboxes[max_i + i + 1] = bboxes[max_i]
        d["bbox"] = torch.stack(bboxes)
        return d
       
class test(Transform):
    def __init__(self, i) -> None:
        super().__init__()
        self.i = i
        
    def __call__(self, data: Any):
        d = dict(data)
        print(self.i, d["3D_mask"].shape)
        return d 
    
class get_3d_single_box(Transform):
    # 直接获取box
    def __init__(self, axis, aug=None, max_bbox=False):
        super().__init__()
        self.axis = axis
        self.aug = aug
        self.max_bbox = max_bbox
    
    def __call__(self, data):
        d = dict(data)
        bboxes = []
        img = d["3D_mask"][0]
        ct_img = d["3D_image"][0]
        if torch.max(img) > 1:
            img[img == 1] = 0
            img[img >= 2] = 1
            d["3D_mask"][0] = img
        exist = []
        for i in range(img.shape[self.axis]):
            slice_img = slice_array(img, self.axis, i)
            if torch.sum(slice_img) == 0:
                bbox_2d = torch.zeros((1, 4))
            else:
                bbox_2d = single_box_generater(slice_img)
                if self.max_bbox:
                    max_area = 0
                    for i in bbox_2d:
                        bbox_area = (i[2]-i[0]) * (i[3]-i[1])
                        if bbox_area > max_area:
                            max_area = bbox_area
                            max_bbox = i
                    bbox_2d = max_bbox[None, :]
                    assert bbox_2d.shape == (1,4), "wrong bbox shape"
                exist.append(i)       
            bboxes.append(bbox_2d)
        if self.aug is not None:
            min_i = exist[0]
            max_i = exist[-1]
            # 向下扩
            img_slice = ct_img[min_i]
            mask_slice = img[min_i]
            mask_value = img_slice[mask_slice == 1]
            begin_value = np.mean(mask_value)
            for i in range(self.aug):
                if min_i - i - 1 < 0:
                    break
                img_slice = ct_img[min_i - i - 1]
                mask_value = img_slice[mask_slice == 1]
                if np.abs(np.mean(mask_value) - begin_value) > 200:
                    break
                bboxes[min_i - i - 1] = bboxes[min_i]
                begin_value = np.mean(mask_value)
            # 向上扩
            img_slice = ct_img[max_i]
            mask_slice = img[max_i]
            mask_value = img_slice[mask_slice == 1]
            begin_value = np.mean(mask_value)
            for i in range(self.aug):
                if max_i + i + 1 > 31:
                    break
                img_slice = ct_img[min_i + i + 1]
                mask_value = img_slice[mask_slice == 1]
                if np.abs(np.mean(mask_value) - begin_value) > 200:
                    break
                bboxes[max_i + i + 1] = bboxes[max_i]
                begin_value = np.mean(mask_value)
        d["bbox"] = bboxes
        return d
    

class unify_spacing(Transform):
    def __init__(self, ori_z) -> None:
        self.ori_z = ori_z
        super().__init__()
    
    def __call__(self, data):
        d = dict(data)
        assert d["3D_image"].shape == (1, self.ori_z, 128, 128),"wrong image shape"
        z_size = int(self.ori_z * d["3D_image"].meta["pixdim"][3]/ d["3D_image"].meta["pixdim"][1])
        img_resize = Resize((z_size, 128, 128), mode="trilinear")
        mask_resize = Resize((z_size, 128, 128), mode="nearest")
        d["3D_image"] = img_resize(d["3D_image"])
        d["3D_mask"] = mask_resize(d["3D_mask"])
        return d
    
        
if __name__ == "__main__":   
    a = torch.ones((1,3,5))
    b = torch.nn.functional.pad(a,(2,2,2,2), "constant", 0)
    print(b[0])
    print(box_generater(b[0]))
    from skimage import measure
    arr = b[0].numpy().astype(int)
    print(arr)
    print(single_box_generater(arr))
        #print(single_box_generater(roi_arr))
    
    
    