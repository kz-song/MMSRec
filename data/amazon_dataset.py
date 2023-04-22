import os
import torch
from data.dataset import SequentialDataset


class AmazonSequentialDataset(SequentialDataset):
    def __init__(self,
                 args,
                 data_part,
                 item_file,
                 seq_file,
                 vision_format,
                 text_format,
                 **kwargs):

        super().__init__(args,
                         data_part,
                         item_file=item_file,
                         seq_file=seq_file,
                         vision_format=vision_format,
                         text_format=text_format,
                         **kwargs)

    def _load_vision_embed(self, path):
        vision = torch.zeros((self.max_vision_frames, self.vision_feature_embed_dim))
        mask = torch.zeros(self.max_vision_frames, dtype=torch.bool)

        if path is not None and os.path.exists(path):
            embeds = torch.load(path)
            embeds.requires_grad_(False)
            vision[0, :] = embeds
            mask[0] = True
        return vision, mask

    def _load_text_embed(self, path):
        text = torch.zeros(self.text_feature_embed_dim)
        mask = torch.zeros(1, dtype=torch.bool)

        if path is not None and os.path.exists(path):
            text = torch.load(path)
            text.requires_grad_(False)
            mask = torch.ones(1, dtype=torch.bool)
        return text, mask

