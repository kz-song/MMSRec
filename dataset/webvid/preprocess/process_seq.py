import sys
sys.path.append("../../../")
import jsonlines
from tqdm import tqdm
from utils.data_utils import load_metas


def save_metas_sequence(metas, outfile):
    with jsonlines.open(outfile, mode='w') as wfile:
        for line in tqdm(metas, desc="write metas"):
            result = [line["id"]]
            wfile.write(result)


if __name__ == "__main__":
    meta_file = '../processed/train_item.jsonl'
    outfile = "../processed/train_seq.jsonl"
    metas = load_metas(meta_file)
    save_metas_sequence(metas, outfile)

    meta_file = '../processed/eval_item.jsonl'
    outfile = "../processed/eval_seq.jsonl"
    metas = load_metas(meta_file)
    save_metas_sequence(metas, outfile)





