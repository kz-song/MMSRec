import sys
sys.path.append("../../../")
import os
import clip
import torch
import chardet
import jsonlines
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import image_transform, load_video
from transformers import BertModel, AutoTokenizer, AutoImageProcessor, ResNetModel


def load_inter_file(inter_file):
    with open(inter_file, "rb") as f:
        encode_type = chardet.detect(f.read())

    inters = []
    with open(inter_file, "r", encoding=encode_type["encoding"]) as f:
        for data in tqdm(f.readlines(), desc=f"load inter file {inter_file}"):
            user, item, rate, time = data.strip().split("::")
            inters.append({"user": user, "item": item, "rate": rate, "time": time})
    return inters


def load_meta_file(meta_file):
    with open(meta_file, "rb") as f:
        encode_type = chardet.detect(f.read())

    metas = {}
    with open(meta_file, "r", encoding=encode_type["encoding"]) as f:
        for line in tqdm(f.readlines(), desc=f"load meta file {meta_file}"):
            id, name, tag = line.strip().split("::")
            metas[id] = {"name": name, "tag": tag}
    return metas


def filter_k_core_inters(inters, user_inter_threshold=5, item_inter_threshold=5):
    print(f"Filter K core: user {user_inter_threshold}, item {item_inter_threshold}")
    while True:
        user_count = {}
        item_count = {}
        for inter in inters:
            if inter["user"] not in user_count:
                user_count[inter["user"]] = 1
            else:
                user_count[inter["user"]] += 1

            if inter["item"] not in item_count:
                item_count[inter["item"]] = 1
            else:
                item_count[inter['item']] += 1

        new_inters = []
        for inter in inters:
            if user_count[inter["user"]] >= user_inter_threshold and \
                    item_count[inter["item"]] >= item_inter_threshold:
                new_inters.append(inter)

        print(f"\tFilter: {len(inters)} inters to {len(new_inters)} inters")
        if len(new_inters) == len(inters):
            return new_inters
        inters = new_inters


def group_inters_by_user(inters):
    users = {}
    for inter in tqdm(inters, desc="group inters by user"):
        if inter["user"] not in users:
            users[inter["user"]] = []
        users[inter["user"]].append({"item": inter["item"], "time": inter["time"]})
    return users


def filter_inters_by_metas(inters, metas):
    new_inters = []
    for inter in tqdm(inters, desc="filter inters by metas"):
        if inter["item"] in metas:
            new_inters.append(inter)
    return new_inters


def filter_metas_by_inters(metas, inters):
    items = set()
    for inter in tqdm(inters, desc="filter metas by inters"):
        items.add(inter["item"])
    new_metas = {}
    for id, meta in metas.items():
        if id in items:
            new_metas[id] = meta
    return new_metas


class MovielensDataset(Dataset):
    def __init__(self, args, model):
        self.args = args
        self.path = args.processed_path

        self.item_file = args.item_file
        self.items = self._load_item_file()

        self.text_feature_path = args.text_feature_path
        self.vision_feature_path = args.vision_feature_path

        self.model = model

    def _load_item_file(self):
        items = []

        file_path = os.path.join(self.path, self.item_file)
        with jsonlines.open(file_path, mode='r') as rfile:
            for line in rfile:
                items.append(line)

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        id = item["id"]
        vision_src_path = "/".join(item["vision"].split("/")[4:]) if item["vision"] else None
        vt, vision = self._load_vision_file(vision_src_path)
        text_src_path = "/".join(item["text"].split("/")[4:]) if item["text"] else None
        tt, text = self._load_text_file(text_src_path)

        vision_tgt_path = os.path.join(self.path, self.vision_feature_path, str(id) + ".pth") if vt else None
        text_tgt_path = os.path.join(self.path, self.text_feature_path, str(id) + ".pth") if tt else None

        return id, vision, vision_tgt_path, text, text_tgt_path

    def _load_vision_file(self, file):
        try:
            vision = load_video(file, 10)
            vision = self.model.vision_process(vision)
            return True, vision
        except:
            return False, torch.zeros((10, 3, 224, 224))

    def _load_text_file(self, file):
        try:
            with open(file, 'r', encoding='utf-8') as rfile:
                text = rfile.readline().strip()
            return True, text
        except:
            return False, ""


class MovielensDataloader(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def _collect_func(self, data):
        id_list = []
        vision_list = []
        vision_tgt_list = []
        text_list = []
        text_tgt_list = []

        for id, vision, vision_tgt, text, text_tgt in data:
            id_list.append(id)
            vision_list.append(vision)
            vision_tgt_list.append(vision_tgt)
            text_list.append(text)
            text_tgt_list.append(text_tgt)

        vision_list = torch.cat(vision_list, dim=0)
        text_list = self.model.text_process(text_list)

        return id_list, vision_list, vision_tgt_list, text_list, text_tgt_list

    def get_dataloader(self, dataset):
        dataloader = DataLoader(dataset,
                                batch_size=100,
                                num_workers=4,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=self._collect_func)
        return dataloader


class ClipModel(object):
    def __init__(self, args):
        self.args = args
        model, preprocess = clip.load(args.clip_model_path, device="cpu")
        self.model = model.to(args.device).eval()

        self.image_processor = image_transform(model.visual.input_resolution)
        self.text_processor = clip

    def vision_process(self, images):
        vision = self.image_processor(images)
        return vision

    def text_process(self, texts):
        tokens = self.text_processor.tokenize(texts, truncate=True)
        return tokens

    def vision_encode(self, vision):
        vision = vision.to(self.args.device)
        features = self.model.encode_image(vision)
        features = features.to("cpu")
        return features

    def text_encode(self, text):
        text = text.to(self.args.device)
        features = self.model.encode_text(text)
        features = features.to("cpu")
        return features


class BertResNetModel(object):
    def __init__(self, args):
        self.args = args

        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-101")
        vision_model = ResNetModel.from_pretrained("microsoft/resnet-101")
        self.vision_model = vision_model.to(args.device).eval()

        self.text_processor = AutoTokenizer.from_pretrained("bert-base-uncased")
        text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_model = text_model.to(args.device).eval()

    def vision_process(self, images):
        vision = self.image_processor(images, return_tensors="pt")["pixel_values"]
        return vision

    def text_process(self, texts):
        tokens = self.text_processor(texts, padding=True, truncation=True, return_tensors="pt")
        return tokens

    def vision_encode(self, vision):
        vision = vision.to(self.args.device)
        features = self.vision_model(vision).pooler_output.squeeze(-1).squeeze(-1)
        features = features.to("cpu")
        return features

    def text_encode(self, text):
        text = text.to(self.args.device)
        features = self.text_model(**text).last_hidden_state[:, 0, :]
        features = features.to("cpu")
        return features





