import os
import clip
import torch
from abc import abstractmethod
from torch.utils.data import Dataset
from utils.data_utils import load_metas, load_video, image_transform


class BasicDataset(Dataset):
    def __init__(self,
                 args,
                 data_part,
                 item_file,
                 seq_file,
                 vision_format,
                 text_format,
                 **kwargs):

        self.args = args
        self.data_part = data_part
        self.vision_format = vision_format
        self.text_format = text_format

        self.item_id = {}
        self.item_data = {}
        self._load_item_data(item_file)
        self.seq_data = []
        self._load_seq_data(seq_file)

        if args.local_rank == 0:
            print(f"Dataset {self.data_part} items : {len(self.item_id)}")
            print(f"Dataset {self.data_part} sequence : {len(self.seq_data)}")

        self.vision_feature_embed_dim = args.vision_feature_embed_dim
        self.text_feature_embed_dim = args.text_feature_embed_dim
        self.max_seq_length = args.max_seq_length
        self.vision_resolution = args.vision_resolution
        self.max_vision_frames = args.max_vision_frames

        self.image_processor = image_transform(self.vision_resolution)
        self.text_tokenizer = clip.tokenize

    def __len__(self):
        return len(self.seq_data)

    def get_item_num(self):
        return len(self.item_data)

    def get_seq_num(self):
        return len(self.seq_data)

    def __getitem__(self, index):
        input_ids = self.seq_data[index]
        input_ids = input_ids[-self.max_seq_length:]
        vision, vision_mask = self._seq_vision_process(input_ids)
        text, text_mask = self._seq_text_process(input_ids)
        input_ids = torch.tensor(input_ids)

        return input_ids, vision, vision_mask, text, text_mask

    def _seq_vision_process(self, input_ids):
        vision_list = []
        mask_list = []
        for item in input_ids:
            vision, mask = self._get_vision(self.item_data[item]['vision'])
            vision_list.append(vision)
            mask_list.append(mask)

        vision_seq = torch.stack(vision_list, dim=0)
        mask_seq = torch.stack(mask_list, dim=0)

        # vision_seq    mp4 format        torch.tensor((seq, max_frames, 3, H, W))
        #               embed format      torch.tensor((seq, max_frames, dim))
        # mask_seq                        torch.tensor((seq, max_frames))
        return vision_seq, mask_seq

    def _seq_text_process(self, input_ids):
        text_list = []
        mask_list = []
        for item in input_ids:
            text, mask = self._get_text(self.item_data[item]['text'])
            text_list.append(text)
            mask_list.append(mask)

        text_seq = torch.stack(text_list, dim=0)
        mask_seq = torch.cat(mask_list, dim=0)

        # text_seq    txt format        torch.tensor((seq, 77))
        #             embed format      torch.tensor((seq, dim))
        # mask_seq                      torch.tensor(seq)
        return text_seq, mask_seq

    def _load_item_data(self, file_path):
        raw_data = load_metas(file_path)
        new_item_id = 3     # 0 : padding, 1 : masking, 2 : prompt
        for line in raw_data:
            self.item_id[line["id"]] = new_item_id
            self.item_data[new_item_id] = {
                "vision": line['vision'] if self.vision_format is not None else None,
                "text": line['text'] if self.text_format is not None else None
            }
            if line["vision"] is None and line["text"] is None:
                raise Exception(f"All Empty Item Error : {line['id']}")
            new_item_id += 1

    def _load_seq_data(self, file_path):
        raw_data = load_metas(file_path)
        for line in raw_data:
            item_exist = True
            for item in line:
                if item not in self.item_id:
                    item_exist = False
                    break
            if item_exist:
                new_line = [self.item_id[item] for item in line]
                self.seq_data.append(new_line)

    def _get_vision(self, path):
        if self.vision_format == "embed":
            vision, mask = self._load_vision_embed(path)
            assert vision.shape == (self.max_vision_frames, self.vision_feature_embed_dim)
            assert mask.shape == (self.max_vision_frames,)
            return vision, mask

        elif self.vision_format == "mp4":
            return self._vision_preprocess(path)

        elif self.vision_format is None:
            vision = torch.zeros((self.max_vision_frames, self.vision_feature_embed_dim))
            mask = torch.zeros(self.max_vision_frames, dtype=torch.bool)
            return vision, mask

        raise Exception("Config vision format error")

    def _get_text(self, path):
        if self.text_format == "embed":
            text, mask = self._load_text_embed(path)
            assert text.shape == (self.text_feature_embed_dim,)
            assert mask.shape == (1,)
            return text, mask

        elif self.text_format == "txt":
            return self._text_preprocess(path)

        elif self.text_format is None:
            text = torch.zeros(self.text_feature_embed_dim)
            mask = torch.zeros(1, dtype=torch.bool)
            return text, mask

        raise Exception("Config text format error")

    def _vision_preprocess(self, path):
        vision = torch.zeros((self.max_vision_frames, 3, self.vision_resolution, self.vision_resolution))
        mask = torch.zeros(self.max_vision_frames, dtype=torch.bool)

        if path is not None and os.path.exists(path):
            img_array = load_video(path, self.max_vision_frames)
            vision[:img_array.shape[0]] = self.image_processor(img_array)
            mask[:img_array.shape[0]] = True

        # vision    torch.tensor((max_frames, 3, H, W))
        # mask      torch.tensor(max_frames)
        return vision, mask

    def _text_preprocess(self, path):
        text = torch.zeros(77, dtype=torch.int)
        mask = torch.zeros(1, dtype=torch.bool)

        if path is not None and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as rfile:
                data = rfile.readline().strip()
                text = self.text_tokenizer(data, truncate=True)
                text = text.squeeze(dim=0)
            mask = torch.ones(1, dtype=torch.bool)

        # text      torch.tensor(77)
        # mask      torch.tensor(1)
        return text, mask

    @abstractmethod
    def _load_vision_embed(self, path):
        # vision      torch.tensor((max_frames, dim))
        # mask        torch.tensor(max_frames)
        raise NotImplementedError

    @abstractmethod
    def _load_text_embed(self, path):
        # text       torch.tensor(dim)
        # mask       torch.tensor(1)
        raise NotImplementedError


class SequentialDataset(BasicDataset):
    def __init__(self, args, data_part, item_file, seq_file, vision_format, text_format, **kwargs):
        super().__init__(args, data_part, item_file, seq_file, vision_format, text_format, **kwargs)

    def __getitem__(self, index):
        input_ids = self.seq_data[index]
        input_ids = input_ids[-self.max_seq_length:]
        input_ids = torch.tensor(input_ids)

        return input_ids

    def full_item_features(self):
        vision_list = []
        vision_mask_list = []
        text_list = []
        text_mask_list = []

        for id in sorted(self.item_data.keys()):
            value = self.item_data[id]
            vision, vision_mask = self._get_vision(value["vision"])
            text, text_mask = self._get_text(value["text"])

            vision_list.append(vision)
            vision_mask_list.append(vision_mask)
            text_list.append(text)
            text_mask_list.append(text_mask)

        full_vision = torch.stack(vision_list, dim=0)
        full_vision_mask = torch.stack(vision_mask_list, dim=0)
        full_text = torch.stack(text_list, dim=0)
        full_text_mask = torch.cat(text_mask_list, dim=0)

        return full_vision, full_vision_mask, full_text, full_text_mask



















