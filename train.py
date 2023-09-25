
import torch
import numpy as np
from monai.data import DataLoader
from monai.data.dataset import CacheDataset
from utils.Transform import get_box, box_jitter, box_zoom
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
from segment_anything import sam_model_registry
from model.medsam_trainer import SAMTrainer
import json
from utils.ddp_utils import init_distributed_mode
from torch.optim.lr_scheduler import CyclicLR
#设置显卡间通信方式
torch.multiprocessing.set_sharing_strategy('file_system') 


# fix random seeds for reproducibility
SEED = 2023
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  #在gpu训练时固定随机源
torch.backends.cudnn.benchmark = True   #搜索卷积方式，启动算法前期较慢，后期会快
# torch.autograd.detect_anomaly() # debug的时候启动，用于检查异常
np.random.seed(SEED)

def main(config):
    
    
    #%% set up transform
    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label", "mask"]),
        EnsureChannelFirstd(keys=["image", "label", "mask"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label", "mask"], dtype=torch.long),
        ScaleIntensityRanged("image", -1250, 500, 0, 1, True),
        RandFlipd(keys=["image", "label", "mask"], prob=0.1, spatial_axis=0),
        RandFlipd(keys=["image", "label", "mask"], prob=0.1, spatial_axis=1),
        RandZoomd(keys=["image", "label", "mask"], prob=0.1, min_zoom=0.5, max_zoom=2,
                  mode=("bilinear", "nearest", "nearest"),
                  padding_mode=("constant", "constant", "constant"),
                  ),
        RandRotated(keys=["image", "label", "mask"], prob=0.1, range_x=(-.3, .3),
            mode=("bilinear", "nearest", "nearest"),
            padding_mode="zeros"),
        RandAffined(keys=["image", "label", "mask"], spatial_size=(128, 128),
                    translate_range=(-50, 50),
                    prob = 0.9,
                    mode=("bilinear", "nearest", "nearest"),
                    padding_mode=("zeros", "zeros", "zeros")),
        get_box(),
        RandLambdad("bbox", box_jitter, prob=0.5),
        RandLambdad("bbox", box_zoom, prob=0.5),
        RepeatChanneld("image", 3),
        ToTensord(keys=["image", "bbox"], dtype=torch.float32),
        ToTensord(keys=["mask", "label"], dtype=torch.long)
        ])
    
    val_transform = Compose([
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label", "mask"]),
        EnsureChannelFirstd(keys=["image", "label", "mask"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label", "mask"], dtype=torch.long),
        ScaleIntensityRanged("image", -1250, 500, 0, 1, True),
        get_box(),
        RepeatChanneld("image", 3),
        ToTensord(keys=["image", "bbox"], dtype=torch.float32),
        ToTensord(keys=["mask", "label"], dtype=torch.long)
        ])
    
    
    trans = []
    for i in train_transform.transforms:
        trans.append(i.__class__.__name__)
        if hasattr(i, 'prob'):
            trans.append(i.prob)
        
    config["trans"] = trans    

    # setup data_loader instances
    
    with open(config["json_path"]) as f:
        data = json.load(f)
        
    train_ds = CacheDataset(
        data=data["train"],
        transform=train_transform,
        cache_rate=1,
        num_workers=8,
        )    

    val_ds = CacheDataset(
        data=data["validation"],
        transform=val_transform,
        cache_rate=1,
        num_workers=8,
        )

    is_ddp = config["is_ddp"]
    if is_ddp:
        init_distributed_mode()

    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        train_dataloader = DataLoader(train_ds,
                        batch_size=config["batch_size"], num_workers=4,
                        sampler=train_sampler, pin_memory=True)
        val_dataloader = DataLoader(val_ds,
                                    batch_size=config["batch_size"], num_workers=4,
                                    sampler=val_sampler, pin_memory=True)
    else:
        train_dataloader = DataLoader(train_ds,
                        batch_size=config["batch_size"], shuffle=True,
                        num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_ds, 
                            batch_size=config["batch_size"], shuffle=True,
                            num_workers=4, pin_memory=True)

    # build model architecture, then print to console
    model = sam_model_registry["vit_b"]()

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer_method = getattr(torch.optim, config["optimizer"]["type"] )
    optimizer = optimizer_method(model.mask_decoder.parameters(), 
                        lr=config["optimizer"]["args"]["lr"], weight_decay=config["optimizer"]["args"]["weight_decay"])
    lr_scheduler = CyclicLR(optimizer, config["optimizer"]["args"]["lr"], 
                            config["optimizer"]["args"]["lr"]*5, 2000, cycle_momentum=False)

    trainer = SAMTrainer(model, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=val_dataloader,
                      valid_interval=config["valid_interval"],
                      lr_scheduler=lr_scheduler,
                    )

    trainer.train()


if __name__ == '__main__':
    config = {
        "json_path": "/data3/home/lishengyong/data/ssc_0802/new_slices/dataset.json",
        "is_ddp": True,
        "resume": "./saved/model/0818_98/checkpoint-epoch15.pth",  #"./saved/model/0807_1536/checkpoint-epoch48.pth"
        "epochs": 100,
        "save_dir":"./saved",
        'arch': "SAM",
        "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 1e-5,
            "weight_decay": 0,
        }},
        "batch_size":4,
        "valid_interval":2
    }
    # ddp module的优化meizuo
    main(config)