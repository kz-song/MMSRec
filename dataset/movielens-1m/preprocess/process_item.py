import sys
sys.path.append('../../../')
import os
import argparse
import jsonlines
from tqdm import tqdm
from general import load_inter_file, load_meta_file, filter_k_core_inters, \
    filter_metas_by_inters, group_inters_by_user


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='../raw', type=str, help='raw data path')
    parser.add_argument('--processed_path', default='../processed', type=str, help='processed data path')

    parser.add_argument('--rate_file', default='ml-1m/ratings.dat', type=str, help='rating meta file')
    parser.add_argument('--meta_file', default='ml-1m/movies.dat', type=str, help='movie meta file')
    parser.add_argument('--video_path', default='videos', type=str, help='downloaded videos path')
    parser.add_argument('--audio_path', default='audios', type=str, help='downloaded audios path')
    parser.add_argument('--text_path', default='texts', type=str, help='downloaded texts path')

    parser.add_argument('--k_core', default=5, type=int, help='filter inters by k core')
    parser.add_argument('--item_outfile', default='item.jsonl', type=str, help='processed items meta file')
    parser.add_argument('--train_seq_outfile', default='train_seq.jsonl', type=str, help='processed seq train file')
    parser.add_argument('--eval_seq_outfile', default='eval_seq.jsonl', type=str, help='processed seq eval file')
    parser.add_argument('--test_seq_outfile', default='test_seq.jsonl', type=str, help='processed seq test file')
    args = parser.parse_args()
    return args


class MovielensProcessor(object):
    def __init__(self, args):
        self.args = args
        self.prefix_path= "./dataset/movielens-1m/preprocess"

        self.raw_path = args.raw_path
        self.processed_path = args.processed_path
        self.video_path = args.video_path
        self.audio_path = args.audio_path
        self.text_path = args.text_path

        self.item_outfile = args.item_outfile
        self.train_seq_outfile = args.train_seq_outfile
        self.eval_seq_outfile = args.eval_seq_outfile
        self.test_seq_outfile = args.test_seq_outfile

    def write_item_file(self, metas):
        print("Process Item Data:")
        process_metas = []
        for id, meta in tqdm(metas.items()):
            video_dir = os.path.join(self.processed_path, self.video_path, id + ".mp4")
            audio_dir = os.path.join(self.processed_path, self.audio_path, id + ".wav")
            text_dir = os.path.join(self.processed_path, self.text_path, id + ".txt")

            process_metas.append({"id": id,
                                  "vision": os.path.join(self.prefix_path, video_dir) if os.path.exists(video_dir) else None,
                                  "audio": os.path.join(self.prefix_path, audio_dir) if os.path.exists(audio_dir) else None,
                                  "text": os.path.join(self.prefix_path, text_dir) if os.path.exists(text_dir) else None})

        with jsonlines.open(os.path.join(self.processed_path, self.item_outfile), mode='w') as wfile:
            for line in tqdm(process_metas, desc="write metas"):
                wfile.write(line)
        print(f"Total Processed {len(process_metas)} items")

    def write_seq_file(self, users):
        print("Process Seq Data:")
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

        train_file = os.path.join(self.processed_path, self.train_seq_outfile)
        with jsonlines.open(train_file, mode='w') as wfile:
            for line in train_seq_data:
                wfile.write(line)

        eval_file = os.path.join(self.processed_path, self.eval_seq_outfile)
        with jsonlines.open(eval_file, mode='w') as wfile:
            for line in eval_seq_data:
                wfile.write(line)

        test_file = os.path.join(self.processed_path, self.test_seq_outfile)
        with jsonlines.open(test_file, mode='w') as wfile:
            for line in test_seq_data:
                wfile.write(line)

    def process(self):
        inters = load_inter_file(os.path.join(self.raw_path, self.args.rate_file))
        metas = load_meta_file(os.path.join(self.raw_path, self.args.meta_file))

        inters = filter_k_core_inters(inters, self.args.k_core, self.args.k_core)
        metas = filter_metas_by_inters(metas, inters)
        self.write_item_file(metas)

        users = group_inters_by_user(inters)
        self.write_seq_file(users)


if __name__ == '__main__':
    args = parse_args()

    pid = MovielensProcessor(args)
    pid.process()







