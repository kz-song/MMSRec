import os
import sys
sys.path.append('../../../')
import argparse
import jsonlines
from tqdm import tqdm


class WebvidProcessor(object):
    def __init__(self, args):
        self.args = args

        self.prefix_path= "./dataset/webvid/preprocess"
        self.raw_path = args.raw_path
        self.processed_path = args.processed_path

        self.train_path = args.train_path
        self.eval_path = args.eval_path
        self.train_outfile = args.train_outfile
        self.eval_outfile = args.eval_outfile

    def _get_meta_paths(self, path):
        sub_paths = [os.path.join(path, sub_path) for sub_path in os.listdir(path)]
        results = [sub_path for sub_path in sub_paths if os.path.isdir(sub_path)]
        return results

    def _get_meta_items(self, path):
        files = [os.path.join(path, file) for file in os.listdir(path)]
        txt_files = [file for file in files if file.endswith(".txt")]
        npy_files = [file for file in files if file.endswith(".npy")]

        npy_dict = {}
        for file in npy_files:
            basename = os.path.basename(file).split(".")[0]
            npy_dict[basename] = file

        metas = []
        for file in tqdm(txt_files, desc="build metas"):
            basename = os.path.basename(file).split(".")[0]
            if basename in npy_dict:
                metas.append({"id": basename,
                              "vision": os.path.join(self.prefix_path, npy_dict[basename]),
                              "text": os.path.join(self.prefix_path, file)})

        return metas

    def _process_items(self, path, outfile):
        meta_paths = self._get_meta_paths(path)

        meta_items = []
        for meta_path in meta_paths:
            print(f"Process path {meta_path}")
            metas = self._get_meta_items(meta_path)
            meta_items.extend(metas)

        with jsonlines.open(outfile, mode='w') as wfile:
            for line in tqdm(meta_items, desc="write metas"):
                wfile.write(line)
        print(f"Total Processed {len(meta_items)} items")

    def process_train_data(self):
        self._process_items(os.path.join(self.raw_path, self.train_path),
                            os.path.join(self.processed_path, self.train_outfile))

    def process_eval_data(self):
        self._process_items(os.path.join(self.raw_path, self.eval_path),
                            os.path.join(self.processed_path, self.eval_outfile))


def parse_args():
    parser = argparse.ArgumentParser(description='webvid meta data processing')
    parser.add_argument('--raw_path', default='../raw', type=str, help='raw data path')
    parser.add_argument('--processed_path', default='../processed', type=str, help='processed data path')

    parser.add_argument('--train_path', default='train', type=str, help='webvid train meta files path')
    parser.add_argument('--eval_path', default='eval', type=str, help='webvid eval meta files path')
    parser.add_argument('--train_outfile', default='train_item.jsonl', type=str, help='processed training items meta file')
    parser.add_argument('--eval_outfile', default='eval_item.jsonl', type=str, help='processed eval items meta file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    api = WebvidProcessor(args)
    api.process_train_data()
    api.process_eval_data()


