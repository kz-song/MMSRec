import os
import numpy
import torch
from data.dataset import BasicDataset


class WebvidDataset(BasicDataset):
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

        if os.path.exists(path):
            embeds = torch.from_numpy(numpy.load(path, allow_pickle=True))
            if embeds.shape[0] > self.max_vision_frames:
                embeds = embeds[:self.max_vision_frames]
            vision[:embeds.shape[0]] = embeds
            mask[:embeds.shape[0]] = True

        return vision, mask

    def _load_text_embed(self, path):
        raise NotImplementedError




