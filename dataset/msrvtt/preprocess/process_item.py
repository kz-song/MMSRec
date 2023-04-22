import os
import sys
sys.path.append('../../../')
import json
import numpy
import pandas
import argparse
import jsonlines
from tqdm import tqdm
from utils.basic_utils import check_dirs


class MsrvttProcessor(object):
    def __init__(self, args):
        self.args = args

        self.prefix_path = "./dataset/msrvtt/preprocess"
        self.raw_path = args.raw_path
        self.processed_path = args.processed_path
        self.eval_outfile = args.eval_outfile
        self.eval_caption_outpath = args.eval_caption_outpath

        self.eval_caption_ids_file = "MSRVTT/high-quality/structured-symlinks/jsfusion_val_caption_idx.pkl"
        self.eval_file = "MSRVTT/high-quality/structured-symlinks/val_list_jsfusion.txt"
        self.annotation_file = "MSRVTT/annotation/MSR_VTT.json"
        self.video_path = "MSRVTT/videos/all"

    def _write_eval_captions(self, meta):
        for video, caption in tqdm(meta.items(), desc="write caption"):
            file_name = os.path.join(self.raw_path, self.eval_caption_outpath, video + ".txt")
            with open(file_name, "w") as f:
                f.write(caption)

    def load_eval_item_captions(self):
        with open(os.path.join(self.raw_path, self.annotation_file), "r", encoding="utf-8") as f:
            annotations = json.load(f)
        df = pandas.DataFrame(annotations["annotations"])

        val_df = pandas.read_csv(os.path.join(self.raw_path, self.eval_file), names = ["video_id"])
        df = df[df["image_id"].isin(val_df["video_id"])]

        metadata = df.groupby(["image_id"])["caption"].apply(list)
        caps = pandas.Series(numpy.load(os.path.join(self.raw_path, self.eval_caption_ids_file), allow_pickle=True))
        new_res = pandas.DataFrame({"caps": metadata, "cap_idx": caps})
        metadata = new_res.apply(lambda x: x['caps'][x['cap_idx']], axis=1)
        metadata = metadata.to_dict()

        return metadata

    def process_eval_items(self):
        eval_id_caption = self.load_eval_item_captions()
        self._write_eval_captions(eval_id_caption)

        with jsonlines.open(os.path.join(self.processed_path, self.eval_outfile), mode='w') as wfile:
            for video in tqdm(eval_id_caption.keys(), desc="write metas"):
                vision_file = os.path.join(self.prefix_path, self.raw_path, self.video_path, video + ".mp4")
                text_file = os.path.join(self.prefix_path, self.raw_path, self.eval_caption_outpath, video + ".txt")

                wfile.write({"id": video, "vision": vision_file, "text": text_file})


def parse_args():
    parser = argparse.ArgumentParser(description='msrvtt meta data processing')
    parser.add_argument('--raw_path', default='../raw', type=str, help='raw data path')
    parser.add_argument('--processed_path', default='../processed', type=str, help='processed data path')

    parser.add_argument('--eval_outfile', default='eval_item.jsonl', type=str, help='processed eval items meta file')
    parser.add_argument('--eval_caption_outpath', default='eval_caption', type=str, help='eval item caption output path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    check_dirs(os.path.join(args.raw_path, args.eval_caption_outpath))

    api = MsrvttProcessor(args)
    api.process_eval_items()

















