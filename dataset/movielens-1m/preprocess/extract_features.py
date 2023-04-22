import os
import torch
import argparse
import jsonlines
from tqdm import tqdm
from general import ClipModel, BertResNetModel, MovielensDataset, MovielensDataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', default='../processed', type=str, help='processed data path')
    parser.add_argument('--extract_model_type', default="clip", type=str, help="clip | bert&resnet")
    parser.add_argument('--clip_model_path', default="../../../weights/clip/ViT-B-32.pt", type=str)

    parser.add_argument('--item_file', default='item.jsonl', type=str)
    parser.add_argument('--item_out_file', default='item_feature.jsonl', type=str)
    parser.add_argument('--text_feature_path', default='text_features', type=str)
    parser.add_argument('--vision_feature_path', default='vision_features', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    args = parser.parse_args()
    return args


def run(args):
    args.device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() else torch.device("cpu")

    if args.extract_model_type == "clip":
        model = ClipModel(args)
    elif args.extract_model == "bert&resnet":
        model = BertResNetModel(args)
    else:
        raise Exception("Extract Model Error!")

    print(f"Processing data----------")
    os.makedirs(os.path.join(args.processed_path, args.vision_feature_path), exist_ok=True)
    os.makedirs(os.path.join(args.processed_path, args.text_feature_path), exist_ok=True)

    dataset = MovielensDataset(args, model)
    dataloader = MovielensDataloader(args, model).get_dataloader(dataset)

    item_metas = []
    for id, vision, vision_path, text, text_path in tqdm(dataloader):
        with torch.no_grad():
            vision_features = model.vision_encode(vision)
            text_features = model.text_encode(text)

        for idx in range(len(id)):
            if vision_path[idx] is not None:
                torch.save(vision_features[idx * 10: (idx + 1) * 10, :].clone().detach(), vision_path[idx])
            if text_path[idx] is not None:
                torch.save(text_features[idx, :].clone().detach(), text_path[idx])

            item_metas.append({"id": id[idx],
                               "vision": os.path.join("./dataset/movielens-1m/preprocess", vision_path[idx]) if vision_path[idx] else None,
                               "audio": None,
                               "text": os.path.join("./dataset/movielens-1m/preprocess", text_path[idx]) if text_path[idx] else None})

    with jsonlines.open(os.path.join(args.processed_path, args.item_out_file), mode="w") as wfile:
        for line in tqdm(item_metas, desc="write info"):
            wfile.write(line)


if __name__ == "__main__":
    args = parse_args()
    run(args)







