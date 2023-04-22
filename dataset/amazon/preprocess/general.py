import os
import gzip
import clip
import torch
import jsonlines
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer, AutoImageProcessor, ResNetModel


def get_sub_paths(path):
    sub_paths = [os.path.join(path, sub_path) for sub_path in os.listdir(path)]
    results = [sub_path for sub_path in sub_paths if os.path.isdir(sub_path)]
    return results


def _parse_csv_line(line):
    data = line.strip().split(",")
    return {"user": str(data[0]), "item": str(data[1]), "rate": float(data[2]), "time": int(data[3])}


def load_inter_file(inter_file):
    assert inter_file.endswith(".csv")

    inters = set()
    with open(inter_file, "r", encoding="utf-8") as fobj:
        for line in tqdm(fobj.readlines(), desc=f"load inter file {inter_file}"):
            data = _parse_csv_line(line)
            user = data["user"]
            item = data["item"]
            rate = data["rate"]
            time = data["time"]
            inters.add((user, item, rate, time))
    print('Total inters:', len(inters))
    return inters


def _parse_gz_line(line):
    return eval(line)


def _parse_vision(data):
    try:
        if "imUrl" in data:
            image_url = data["imUrl"]
            if image_url:
                return image_url
        return None
    except:
        return None


def _parse_text(data):
    try:
        text = ""
        if "title" in data:
            text += f"Title: {str(data['title'])} ; "
        if "description" in data:
            text += f"Description: {str(data['description'])} ; "

        if text:
            return text
        else:
            return f"Asin: {str(data['asin'])} ; "
    except:
        return None


def load_meta_file(meta_file):
    assert meta_file.endswith(".json.gz")

    metas = {}
    gzip_file = gzip.open(meta_file, 'r')
    for line in tqdm(gzip_file, desc=f"load meta file {meta_file}"):
        data = _parse_gz_line(line)
        item = str(data["asin"])

        vision = _parse_vision(data)
        text = _parse_text(data)

        metas[item] = {"vision": vision, "audio": None, "text": text}
    return metas


def filter_inters_by_metas(inters, metas):
    new_inters = []
    for inter in tqdm(inters, desc="filter inters by metas"):
        if inter[1] in metas:
            new_inters.append(inter)
    return new_inters


def filter_metas_by_inters(metas, inters):
    items = set()
    for inter in tqdm(inters, desc="filter metas by inters"):
        items.add(inter[1])
    new_metas = {}
    for id, meta in metas.items():
        if id in items:
            new_metas[id] = meta
    return new_metas


def filter_k_core_inters(inters, user_inter_threshold=5, item_inter_threshold=5):
    print(f"Filter K core: user {user_inter_threshold}, item {item_inter_threshold}")
    while True:
        user_count = {}
        item_count = {}
        for inter in inters:
            if inter[0] not in user_count:
                user_count[inter[0]] = 1
            else:
                user_count[inter[0]] += 1

            if inter[1] not in item_count:
                item_count[inter[1]] = 1
            else:
                item_count[inter[1]] += 1

        new_inters = []
        for inter in inters:
            if user_count[inter[0]] >= user_inter_threshold and \
                    item_count[inter[1]] >= item_inter_threshold:
                new_inters.append(inter)

        print(f"\tFilter: {len(inters)} inters to {len(new_inters)} inters")
        if len(new_inters) == len(inters):
            return new_inters
        inters = new_inters


def group_inters_by_user(inters):
    users = {}
    for inter in tqdm(inters, desc="group inters by user"):
        if inter[0] not in users:
            users[inter[0]] = []
        users[inter[0]].append({"item": inter[1], "time": inter[3]})
    return users


def filter_metas_without_modality(metas, vision_filter=False, text_filter=False):
    desc = "filter metas"
    desc += " [without vision]" if vision_filter else ""
    desc += " [without text]" if text_filter else ""

    new_metas = {}
    for item, meta in tqdm(metas.items(), desc=desc):
        if vision_filter and meta["vision"] is None:
            continue
        if text_filter and meta["text"] is None:
            continue
        new_metas[item] = meta
    return new_metas


class AmazonDataset(Dataset):
    def __init__(self, args, model, path):
        self.args = args
        self.path = path

        self.item_file = args.item_file
        self.items = self._load_item_file()

        self.text_feature_path = args.text_feature_path
        self.vision_feature_path = args.vision_feature_path

        self.model = model

    def _load_item_file(self):
        items = []

        item_file = os.path.join(self.path, self.item_file)
        with jsonlines.open(item_file, mode='r') as rfile:
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
            vision = self.model.vision_process(file)
            return True, vision
        except:
            return False, torch.zeros((3, 224, 224))

    def _load_text_file(self, file):
        try:
            with open(file, 'r', encoding='utf-8') as rfile:
                text = rfile.readline().strip()
            return True, text
        except:
            return False, ""


class AmazonDataloader(object):
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

        vision_list = torch.stack(vision_list, dim=0)
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

        self.image_processor = preprocess
        self.text_processor = clip

    def vision_process(self, file):
        vision = self.image_processor(Image.open(file))
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

    def vision_process(self, file):
        vision = self.image_processor(Image.open(file), return_tensors="pt")["pixel_values"].squeeze(0)
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


