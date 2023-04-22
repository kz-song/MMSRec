import os
import argparse
import jsonlines
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from general import get_sub_paths, load_inter_file, load_meta_file, filter_inters_by_metas, filter_metas_by_inters,\
    filter_k_core_inters, group_inters_by_user, filter_metas_without_modality


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='../raw', type=str, help='raw data path')
    parser.add_argument('--processed_path', default='../processed', type=str, help='processed data path')

    parser.add_argument('--k_core', default=5, type=int, help='filter inters by k core')
    parser.add_argument('--vision_filter', default=False, type=bool, help='Throw items without vision. Default False')
    parser.add_argument('--text_filter', default=False, type=bool, help='Throw items without text. Default False')

    parser.add_argument('--item_outfile', default='item.jsonl', type=str, help='processed items meta file')
    parser.add_argument('--train_seq_outfile', default='train_seq.jsonl', type=str, help='processed train seq file')
    parser.add_argument('--eval_seq_outfile', default='eval_seq.jsonl', type=str, help='processed eval seq file')
    parser.add_argument('--test_seq_outfile', default='test_seq.jsonl', type=str, help='processed test seq file')
    parser.add_argument('--text_outpath', default='texts', type=str, help='text raw files')
    parser.add_argument('--vision_outpath', default='visions', type=str, help='vision raw files')
    args = parser.parse_args()
    return args


class AmazonProcessor(object):
    def __init__(self, args):
        self.args = args

        self.prefix_path = "./dataset/amazon/preprocess"

        self.raw_path = args.raw_path
        self.processed_path = args.processed_path

        self.target_item_file = args.item_outfile
        self.target_train_seq_file = args.train_seq_outfile
        self.target_eval_seq_file = args.eval_seq_outfile
        self.target_test_seq_file = args.test_seq_outfile

        self.text_out_path = args.text_outpath
        self.vision_out_path = args.vision_outpath

        self.sub_paths = get_sub_paths(self.raw_path)

    def requestPicture(self, image_url, save_image_path):
        headers = {'Connection': 'close'}
        try:
            with requests.get(url=image_url, headers=headers) as request_result:
                if request_result.status_code == 200:
                    with open(save_image_path, 'wb') as fileObj:
                        fileObj.write(request_result.content)
                    return True
        except Exception:
            return False

    def multiple_process_item(self, id, meta, path):
        text = meta["text"]
        image = meta["vision"]

        text_file = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.text_out_path, f"{str(id)}.txt")
        if not os.path.exists(text_file):
            with open(text_file, "w", encoding="utf-8") as fobj:
                fobj.write(text)
        text_path = os.path.join(self.prefix_path, text_file)

        image_path = None
        if image is not None:
            image_file = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.vision_out_path, f"{str(id)}{os.path.splitext(image)[-1]}")
            if os.path.exists(image_file) or self.requestPicture(image, image_file):
                image_path = os.path.join(self.prefix_path, image_file)
        return id, image_path, text_path

    def process_item_data(self, metas, path):
        text_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.text_out_path)
        os.makedirs(text_path, exist_ok=True)
        vision_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.vision_out_path)
        os.makedirs(vision_path, exist_ok=True)

        print(f"Process Item Data: {len(metas)}")
        new_metas = {}
        with ThreadPoolExecutor(max_workers=128) as executor:
            process_list = []

            for id, meta in metas.items():
                process = executor.submit(self.multiple_process_item, id, meta, path)
                process_list.append(process)

            for process in tqdm(as_completed(process_list), total=len(process_list)):
                id, image_path, text_path = process.result()
                new_metas[id] = {"vision": image_path, "audio": None, "text": text_path}

        return new_metas

    def write_item_file(self, metas, path):
        item_data = []
        for id, meta in tqdm(metas.items(), desc="write item file"):
            item_data.append({"id": id, "vision": meta["vision"], "audio": None, "text": meta["text"]})

        item_file = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.target_item_file)
        with jsonlines.open(item_file, mode='w') as wfile:
            for line in item_data:
                wfile.write(line)

    def write_seq_file(self, users, path):
        print(f"Process Seq Data: {len(users)}")
        train_seq_data = []
        eval_seq_data = []
        test_seq_data = []
        for id, interacts in tqdm(users.items()):
            interacts = sorted(interacts, key=lambda item: item["time"])
            interacts = [item["item"] for item in interacts]

            for index in range(2, len(interacts) - 1):
                train_seq_data.append(interacts[:index])
            eval_seq_data.append(interacts[:-1])
            test_seq_data.append(interacts[:])

        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        os.makedirs(target_path, exist_ok=True)

        train_file = os.path.join(target_path, self.target_train_seq_file)
        with jsonlines.open(train_file, mode='w') as wfile:
            for line in train_seq_data:
                wfile.write(line)

        eval_file = os.path.join(target_path, self.target_eval_seq_file)
        with jsonlines.open(eval_file, mode='w') as wfile:
            for line in eval_seq_data:
                wfile.write(line)

        test_file = os.path.join(target_path, self.target_test_seq_file)
        with jsonlines.open(test_file, mode='w') as wfile:
            for line in test_seq_data:
                wfile.write(line)

    def _get_raw_file(self, path):
        raw_files = os.listdir(path)
        inter_file = [file for file in raw_files if file.endswith(".csv")][0]
        inter_file = os.path.join(path, inter_file)
        meta_file = [file for file in raw_files if file.startswith("meta") and file.endswith(".json.gz")][0]
        meta_file = os.path.join(path, meta_file)
        return inter_file, meta_file

    def process(self):
        for path in self.sub_paths:
            print(f"\n-----Processing data {path}")
            inter_file, meta_file = self._get_raw_file(path)

            inters = load_inter_file(inter_file)
            metas = load_meta_file(meta_file)

            inters = filter_k_core_inters(inters, self.args.k_core, self.args.k_core)
            metas = filter_metas_by_inters(metas, inters)
            metas = self.process_item_data(metas, path)
            metas = filter_metas_without_modality(metas, self.args.vision_filter, self.args.text_filter)

            inters = filter_inters_by_metas(inters, metas)
            inters = filter_k_core_inters(inters, self.args.k_core, self.args.k_core)
            users = group_inters_by_user(inters)
            self.write_seq_file(users, path)

            metas = filter_metas_by_inters(metas, inters)
            self.write_item_file(metas, path)


if __name__ == '__main__':
    args = parse_args()

    api = AmazonProcessor(args)
    api.process()




