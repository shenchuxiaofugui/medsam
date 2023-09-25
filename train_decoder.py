# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
# from torch.utils.data import DataLoader
from monai.data import DataLoader
import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR
from utils.utils import EarlyStopping
from monai.metrics import DiceMetric
from model.med_sam import prompter_network
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    RandLambdad,
    ToTensord,
)
from utils.Transform import box_jitter, Loadh5
from monai.data.dataset import CacheDataset
from pathlib import Path
from torch.utils.data.sampler import WeightedRandomSampler

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr_npy_path', type=str, default='/data3/home/lishengyong/data/ssc/2D_bbox/processed_data/zoom_rotate_oritrain/img_embadding',
                    help='path to training npy files; two subfolders: npy_gts and npy_embs')
parser.add_argument('-val', '--val_npy_path', type=str, default='/data3/home/lishengyong/data/ssc/2D_bbox/processed_data/zoom_rotate_val/img_embadding',
                    help='path to training npy files; two subfolders: npy_gts and npy_embs')
parser.add_argument('--task_name', type=str, default='new/ssc_new_sampler')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='./work_dir')
parser.add_argument('--runs', type=str, default="./runs/new/ssc_new_sampler", help="tensorboard runs")
# train
parser.add_argument('--num_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--dataset', type=str, default="load_one_by_one")
args = parser.parse_args()





# %% set up model for fine-tuning 
# init tensorboard
my_writer = SummaryWriter(log_dir=args.runs)
early_stopping = EarlyStopping(20)
device = args.device
model_save_path = join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)


# Set up the optimizer, hyperparameter tuning will improve performance here
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

#%% train


# load trained model
num_epochs = args.num_epochs
checkpoint = args.checkpoint
steped = 0
best_metric = 0
best_epoch = 0
if os.path.exists(join(model_save_path, 'run_info.pth')):
    model_info = torch.load(join(model_save_path, 'run_info.pth'))
    steped = model_info["epoch"]
    best_metric = model_info["best_dice"]
    checkpoint = join(model_save_path, "sam_model_latest.pth")
    print(steped, best_metric, checkpoint)
    
sam_model = sam_model_registry[args.model_type](checkpoint=checkpoint).to(device)
sam_prompt = prompter_network(sam_model, True).to(device)
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# set up dataset
train_transform = Compose([
                    Loadh5(),
                    RandLambdad("bbox", box_jitter, prob=0.1),
                    ToTensord("label", torch.long),
                    ToTensord(["img_embedding", "bbox"], torch.float32)
                ])
val_transform = Compose([
                    Loadh5(),
                    ToTensord("label", torch.long),
                    ToTensord(["img_embedding", "bbox"], torch.float32)
                ])
# embedding_ds = getattr(dataset, args.dataset)
ori_files = [i for i in Path(args.tr_npy_path).iterdir()]
aug_files = [i for i in Path(args.tr_npy_path.replace("ori", "")).iterdir()]
val_files = [i for i in Path(args.val_npy_path).iterdir()]
train_dataset = CacheDataset(ori_files, transform=train_transform, num_workers=4, cache_rate=0.1) + \
                CacheDataset(aug_files, transform=train_transform, num_workers=4, cache_rate=0.1)
train_samper = WeightedRandomSampler([5]*len(ori_files)+[1]*len(aug_files), 2*len(ori_files), False)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_samper)
val_dataset = CacheDataset(val_files, transform=val_transform, num_workers=8, cache_rate=0.1)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
sam_prompt = prompter_network(sam_model, True).to(device)
dice_metric = DiceMetric(include_background=False, reduction="mean")
for epoch in range(steped, num_epochs):
    sam_model.train()
    epoch_loss = 0
    # Just train on the first 20 examples
    for step, batch_data in enumerate(tqdm(train_dataloader)):       
        
        image_embedding, gt2D, bbox = (
            batch_data["img_embedding"].to(device),
            batch_data["label"].to(device), batch_data["bbox"].to(device)
            )
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_prompt(bbox, None)
        # do not compute gradients for image encoder and prompt encoder
        with torch.cuda.amp.autocast():
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            ori_res_masks = F.interpolate(
            low_res_masks,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
            )
            loss = seg_loss(ori_res_masks, gt2D)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    epoch_loss /= step
    my_writer.add_scalar("loss", epoch_loss, epoch)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    early_stopping(epoch_loss)
    if early_stopping.early_stop:
        break
    # save the model checkpoint
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))
    torch.save({"epoch": epoch+1, "best_epoch": best_epoch, "best_dice": best_metric}, join(model_save_path, 'run_info.pth'))
    # save the best model
    # if epoch_loss < best_loss:
    #     best_loss = epoch_loss
    #     torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))

    # %% validation
    if (epoch+1) % 5 == 0:
        sam_model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(val_dataloader)):       
        
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
                medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
                ori_res_masks = F.interpolate(
                        medsam_seg_prob,
                        size=(128, 128),
                        mode="bilinear",
                        align_corners=False,
                        )
        # convert soft mask to hard mask
                med_seg = (ori_res_masks > 0.5)
                dice_metric(y_pred=med_seg, y=gt2D)
# convert soft mask to hard mask

            my_writer.add_scalars("pred_dice", {"prediction_dice": dice_metric.aggregate().item()}, epoch+1)
            if dice_metric.aggregate().item() > best_metric:
                torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_val_best.pth'))
                best_metric = dice_metric.aggregate().item()
                best_epoch = epoch
                torch.save({"epoch": epoch+1, "best_epoch": best_epoch, "best_dice": best_metric}, join(model_save_path, 'run_info.pth'))
                print(best_metric)
            dice_metric.reset()
            

                
        

