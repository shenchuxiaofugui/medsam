import numpy as np
import torch
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from monai.metrics import DiceMetric
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from skimage.color import label2rgb
from monai.losses import DiceCELoss


class SAMTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, optimizer, config, 
                 data_loader, valid_data_loader=None, valid_interval=5, lr_scheduler=None):
        super().__init__(model, optimizer, config, data_loader, valid_data_loader, valid_interval)
        self.lr_scheduler = lr_scheduler
        self.criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.train_metrics = DiceMetric(include_background=False, reduction="mean")
        self.valid_metrics = DiceMetric(include_background=False, reduction="mean")
        self.sam_trans = ResizeLongestSide(self.model.image_encoder.img_size)
        self.ori_img_size = (128, 128)
        self.writer_step = 20
        
        if config["resume"] is None:
            checkpoint = torch.load("./work_dir/SAM/medsam_vit_b.pth")
            self.model.load_state_dict(checkpoint)
        
        model = torch.compile(model, mode="default") #pytorch2.0全新特性，类似静态图
        
    
    def _train_batch_step(self, batch_data, batch_idx, epoch):
        img2D, gt2D, bbox = (
                batch_data["image"],
                batch_data["label"],
                batch_data["bbox"].to(self.device),
            )
        if dist.get_rank() == 0 and batch_idx % self.writer_step == 0:
            show_img = img2D[0,0,...]
            show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
            show_img = label2rgb(gt2D[0,0,...].numpy(), show_img.numpy())
            self.writer.add_image_with_boxes("train_gt", show_img,
                            bbox[0][None,:],epoch* (self.len_epoch // self.writer_step)+(batch_idx // self.writer_step), 
                            dataformats="HWC", thickness=1)
        img2D = img2D.to(self.device)  
        gt2D = gt2D.to(self.device)      
                
        # resize input
        img_1024 = F.interpolate(
            img2D,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
            )            
        # img and prompt encoder
        with torch.no_grad():
            image_embedding = self.model.module.image_encoder(img_1024)
            bbox_torch = self.sam_trans.apply_boxes_torch(bbox, self.ori_img_size)
            sparse_embeddings, dense_embeddings = self.model.module.prompt_encoder(
            points=None,
            boxes=bbox_torch,
            masks=None,
        )
            
        # train decoder
        with torch.cuda.amp.autocast():
            low_res_masks, _ = self.model.module.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=self.model.module.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            ori_res_masks = F.interpolate(
            low_res_masks,
            size=self.ori_img_size,
            mode="bilinear",
            align_corners=False,
            )
            loss = self.criterion(ori_res_masks, gt2D)
            # 这里没必要取消梯度缓存，因为encoder根本没算梯度。。。
            # loss = torch.utils.checkpoint.checkpoint(self.criterion,loss)
            
        # train metrics
            med_seg = torch.sigmoid(ori_res_masks)
            med_seg = med_seg > 0.5
            self.train_metrics(y_pred=med_seg, y=gt2D)
            if dist.get_rank() == 0 and batch_idx % self.writer_step == 0:
                show_img = img2D[0,0,...]
                show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
                show_med_seg = med_seg.int()
                show_img = label2rgb(show_med_seg[0,0,...].cpu().numpy(), show_img.cpu().numpy())
                self.writer.add_image_with_boxes("train_pred", show_img,
                        bbox[0][None,:],
                        epoch * (self.len_epoch // self.writer_step)+(batch_idx // self.writer_step), 
                        dataformats="HWC", thickness=1)    
        return loss
  
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        epoch_loss = 0
        self.train_metrics.reset()
        self.model.train()
        for batch_idx, batch_data in enumerate(tqdm(self.data_loader)):
            loss = self._train_batch_step(batch_data, batch_idx, epoch)
            epoch_loss = self._train_batch_end(loss, epoch_loss)

        epoch_loss /= batch_idx
        
        # 看情况在哪更新
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
            # current_lr = self.optimizer.learning_rate.numpy()
            # self.writer.add_scalar("Learning Rate", current_lr, step=epoch)
            
        log = {"train_loss": epoch_loss, "train_dice": self.train_metrics.aggregate().item()}

        return log
    
    def _valid_batch_step(self, batch_data, batch_idx, epoch, epoch_loss):
        show_index = epoch % self.writer_step
        img2D, gt2D, bbox = (
                            batch_data["image"].to(self.device),
                            batch_data["label"].to(self.device),
                            batch_data["bbox"].to(self.device),
                        )
        img_1024 = F.interpolate(
            img2D,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
            ) 
        
        # img and prompt encoder
        with torch.no_grad():
            image_embedding = self.model.module.image_encoder(img_1024)
            bbox_torch = self.sam_trans.apply_boxes_torch(bbox, self.ori_img_size)
            sparse_embeddings, dense_embeddings = self.model.module.prompt_encoder(
            points=None,
            boxes=bbox_torch,
            masks=None,
            )
            

            low_res_masks, _ = self.model.module.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=self.model.module.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            ori_res_masks = F.interpolate(
            low_res_masks,
            size=self.ori_img_size,
            mode="bilinear",
            align_corners=False,
            )
            loss = self.criterion(ori_res_masks, gt2D)
            
        # validation metrics
            med_seg = torch.sigmoid(ori_res_masks)
            med_seg = med_seg > 0.5
            self.valid_metrics(y_pred=med_seg, y=gt2D)
            epoch_loss += loss.item()
            if dist.get_rank() == 0 and (batch_idx+show_index) % self.writer_step == 0:
                show_img = img2D[0,0,...]
                show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
                show_med_seg = med_seg.int()
                show_pred_img = label2rgb(show_med_seg[0,0,...].cpu().numpy(), show_img.cpu().numpy())
                self.writer.add_image_with_boxes("test_pred", show_pred_img,
                                bbox[0][None,:], #batch changdu
                        epoch* (self.len_epoch // self.writer_step)+((batch_idx+show_index) // self.writer_step),
                                dataformats="HWC", thickness=1)

                show_med_seg = gt2D.int()
                show_gt_img = label2rgb(show_med_seg[0,0,...].cpu().numpy(), show_img.cpu().numpy())
                self.writer.add_image_with_boxes("test_gt", show_gt_img,
                                bbox[0][None,:],
                        epoch* (self.len_epoch // self.writer_step)+((batch_idx+show_index) // self.writer_step),
                                dataformats="HWC", thickness=1)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        epoch_loss = 0
        for batch_idx, batch_data in enumerate(tqdm(self.valid_data_loader)):
            epoch_loss = self._valid_batch_step(self, batch_data, batch_idx, epoch, epoch_loss)
            
        # 按lighting再改改        
        epoch_loss /= batch_idx
        log = {"loss": epoch_loss, "dice": self.valid_metrics.aggregate().item()}
        return log
    

