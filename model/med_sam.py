import torch
import torch.nn as nn
from segment_anything.utils.transforms import ResizeLongestSide


class prompter_network(nn.Module):
    def __init__(self, sam_model, boxes=True, ori_size=(128,128)) -> None:
        super().__init__()
        self.sam_model = sam_model
        self.sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.is_box = boxes
        self.ori_size = ori_size


    def forward(self, boxes, mask2D):
        if self.is_box:
            box_torch = self.sam_trans.apply_boxes_torch(boxes, self.ori_size)
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        else:
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=mask2D,
            )
        return sparse_embeddings, dense_embeddings