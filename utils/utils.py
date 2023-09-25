import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt
import time
import os
import json
from pathlib import Path
from skimage import morphology

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=30, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=0.3))
    
def crop2(ori_mask, img_arr, clip_min, clip_max):
    '''
    mask: crop1 mask
    ori_mask: no crop mask
    '''
    ccs1 = measure.regionprops(measure.label(ori_mask))
    new_mask = np.zeros((64, 128, 128))
    for cc in ccs1:
        # 对原始图像的每个连通区域进行遍历
        #clip
        img_elemental = img_arr[tuple(cc.coords.T)]
        img_used = sorted(img_elemental)[len(img_elemental) // 50:]
        a = np.zeros((64, 128, 128), np.int8)
        a[tuple(cc.coords.T)] = 1
        crop_mask_arr = a & (img_arr > clip_min) # (img_arr < clip_max) & 
        # else:
        #     new_mask[tuple(cc.coords.T)] = 1
        #     continue
        
        #如果改连通区域裁没了，还是保留吧
        if np.sum(crop_mask_arr) == 0:
            new_mask[tuple(cc.coords.T)] = 1
            continue
            
            
        # 保留最大的连通区域
        single_cc = measure.label(crop_mask_arr)
        ccs2 = measure.regionprops(single_cc)
        max_area = 0
        for cc2 in ccs2:
            if cc2.area > max_area:
                max_area = cc2.area
                coords = tuple(cc2.coords.T)
        new_mask[coords] = 1
    return new_mask


def read_nii(img_path):
    img = nib.load(img_path)
    return img.get_fdata()

def get_max_cc(img_arr):
    labeled_img = measure.label(img_arr)
    ccs = measure.regionprops(labeled_img)
    max_area = 0
    new_mask = np.zeros((128,128))
    for cc in ccs:
        if cc.area > max_area:
            max_area = cc.area
            coords = tuple(cc.coords.T)
    if len(ccs) > 0:
        new_mask[coords] = 1
    return np.array(new_mask, bool)

def get_slice(case, i, size, mask_crop, axis):
    # 获取同一个slice不同bbox的集合
    slice_file = case / f"{i}_0.nii.gz"
    if not slice_file.exists():
        return np.zeros(size)
    else:
        single_arr = read_nii(slice_file)
        slice_bbox = single_arr == 1
        slice_mask = single_arr == 2
        slice_mask = get_max_cc(slice_mask)
        for number in range(1, 16):
            slice_file = case / f"{i}_{number}.nii.gz"
            if slice_file.exists():
                single_arr = read_nii(slice_file)
                slice_bbox = np.logical_or(slice_bbox, single_arr == 1)
                other_mask = single_arr == 2
                slice_mask = np.logical_or(slice_mask, get_max_cc(other_mask))
            else:
                break
        slice_arr = np.zeros((128, 128))
        slice_arr[slice_bbox] = 1
        if axis == 0:
            slice_arr[slice_mask] = 2
            return slice_arr
        one_box = mask_crop(slice_arr[None,...])[0].numpy()
        one_box = resize(one_box, size, order=0, mode="constant")
        mask_arr = np.zeros((128, 128))
        mask_arr[slice_mask] = 2
        one_mask = mask_crop(mask_arr[None,...])[0].numpy()
        one_mask = resize(one_mask, size, order=0, mode="constant")
        one_box[one_mask != 0] = 2
        return one_box
    
    
def timer(func):
    def func_in():
        start_time = time.time()
        func()
        end_time = time.time()
        spend_time = (end_time - start_time)/60
        print("Spend_time:{} min".format(spend_time))
    return func_in


def generate_json(slice_store_path, is_3d=True):
    if is_3d:
        flag = "3D_"
        json_name = "/dataset_3d.json"
    else:
        flag = ""
        json_name = "/dataset.json"
    cases = os.listdir(slice_store_path+f"/{flag}images")
    cases.sort()
    test_list = []
    for i in cases:
        single_dict = {
            f"{flag}image": slice_store_path+f"/{flag}images/{i}",
            #"label": slice_store_path+f"/labels/{i}",
            f"{flag}mask": slice_store_path+f"/{flag}masks/{i}"
        }
        test_list.append(single_dict)
    print(len(test_list))
    data = {"test":test_list}
    d = json.dumps(data)
    with open(os.path.join(slice_store_path, json_name), "w") as f:
        f.write(d)
        
def opening(mask_path, store_path):
    for i in Path(mask_path).iterdir():
        image = nib.load(i).get_fdata()
        image[image == 1] = 0
        image[image == 2] = 1
        # 定义结构元素
        selem = morphology.disk(1)  # 使用半径为1的圆形结构元素
        # 进行开运算操作
        open_image_list = []
        for slice in image:
            if np.sum(slice) < 20:
                open_image_list.append(slice)
            else:
                open_image_list.append(morphology.opening(slice, selem))
        opened_image = np.stack(open_image_list)
        nib.save(nib.Nifti1Image(opened_image, None), join(store_path, i.name))