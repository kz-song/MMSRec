import os
from data.dataset import BasicDataset


class MSRVTTDataset(BasicDataset):
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
        raise NotImplementedError

    def _load_text_embed(self, path):
        raise NotImplementedError

