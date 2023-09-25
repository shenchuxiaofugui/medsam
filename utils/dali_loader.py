from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import os
import numpy as np
from random import shuffle
from monai.transforms import (
    Compose,
    RandLambdad,
    EnsureChannelFirstd,
    LoadImaged,
    EnsureTyped,
    ScaleIntensityRanged,
    ToTensord,
    RandZoomd,
    RandRotated,
    RandAffined,
    RepeatChanneld,
    Resized,
    RandFlipd,
)
from monai.data.dataset import CacheDataset
from monai.data.dataloader import DataLoader
import torch
from nvidia.dali.backend_impl import TensorGPU
import time
import json




# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra

class ExternalInputIterator(object):
    def __init__(self, batch_size, ds):
        self.ds = ds
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = len(self.ds)
        return self

    def __next__(self):
        if self.i >= self.n:
             self.__iter__()
             raise StopIteration
        leave_num = self.n - self.i
        current_batch_size = min(self.batch_size, leave_num)
        batch, labels = [], []
        for _ in range(current_batch_size):
            batch.append(self.ds[self.i]["image"].as_tensor())
            labels.append(self.ds[self.i]["label"].as_tensor())
            self.i = self.i + 1
        return (batch, labels)

def get_ds():
    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label", "mask"]),
        EnsureChannelFirstd(keys=["image", "label", "mask"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
        ScaleIntensityRanged("image", -1250, 500, 0, 1, True),

        RepeatChanneld("image", 3),
        ToTensord(keys=["image"], dtype=torch.float32),
        ToTensord(keys=["label", "mask"], dtype=torch.long)
        ])
    with open("/data3/home/lishengyong/data/ssc_0802/new_slices/dataset.json") as f:
        data = json.load(f)
        
    train_ds = CacheDataset(
        data=data["train"][:500],
        transform=train_transform,
        cache_rate=0,
        num_workers=8,
        )  
    return train_ds
    


class CustomizeInputGpuIterator(object):
    def __init__(self, images_dir, batch_size):
        self.batch_size = batch_size
        self.files = np.load(images_dir)
        self.n = self.files["gts"].shape[0]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.n:
             self.__iter__()
             raise StopIteration
        leave_num = self.n - self.idx
        current_batch_size = min(self.batch_size, leave_num)
        image_embeddings, gts, sparse_embeddings, dense_embeddings = [], [], [], []
        for _ in range(current_batch_size):
            image_embeddings.append(self.files["img_embeddings"][self.idx])
            gts.append(self.files["gts"][self.idx])
            sparse_embeddings.append(self.files["sparse_embeddings"][self.idx])
            dense_embeddings.append(self.files["dense_embeddings"][self.idx])
            self.idx = self.idx + 1
        return image_embeddings, gts, sparse_embeddings, dense_embeddings
    next = __next__

 # dataloader
class DALI_sscloader(DALIGenericIterator):
    def __init__(self, datapath, batch_size, size=-1, reader_name=None, auto_reset=False, fill_last_batch=None, dynamic_shape=False, last_batch_padded=False, last_batch_policy=LastBatchPolicy.FILL, prepare_first_batch=True):
        eii_gpu = CustomizeInputGpuIterator(datapath, batch_size)

        self.pipe_gpu = Pipeline(batch_size=batch_size, num_threads=8, device_id=0)
        with self.pipe_gpu:
            image_embeddings, gts, sparse_embeddings, dense_embeddings = fn.external_source(source=eii_gpu, num_outputs=4, device="gpu")
        self.pipe_gpu.set_outputs(image_embeddings, gts, sparse_embeddings, dense_embeddings)
        self.pipe_gpu.build()
        super().__init__([self.pipe_gpu], ['image_embedding', 'gt2D', 'sparse_embeddings', 'dense_embeddings'], size, reader_name, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded, last_batch_policy, prepare_first_batch)

class MyDALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(MyDALIPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.ExternalSource()
        fn.external_source
        

    def define_graph(self):
        inputs = self.input()
        return inputs

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def iterable_pipeline():
    jpegs, labels = fn.external_source(source=ExternalInputIterator(batch_size), num_outputs=2,
                                       dtype=[types.UINT8, types.INT32])
    decode = fn.decoders.image(jpegs, device="mixed")
    return decode, labels    

if __name__ == "__main__":
    
    batch_size = 4
    num_threads = 4
    device_id = 0
    eii = ExternalInputIterator(batch_size, get_ds())
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=0)
    with pipe:
        images, labels = fn.external_source(source=eii, num_outputs=2)
        pipe.set_outputs(images, labels)
    pipe.build()
        
    train_data = DALIGenericIterator([pipe], ["image", "label"])
    train_dataloader = DataLoader(get_ds(), 8, batch_size=4)
    start = time.time()
    for i, data in enumerate(train_dataloader):
        img, label = (data["image"].to("cuda:0"), data["label"].to("cuda:0"))
        pass
    end = time.time()
    print(end - start)
    