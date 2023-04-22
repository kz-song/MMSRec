import os
import torch
import argparse
import jsonlines
from tqdm import tqdm
from general import ClipModel, BertResNetModel, get_sub_paths, AmazonDataset, AmazonDataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', default='../processed', type=str)
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

    sub_paths = get_sub_paths(args.processed_path)

    for path in sub_paths:
        print(f"Processing data {path}----------")
        os.makedirs(os.path.join(path, args.vision_feature_path), exist_ok=True)
        os.makedirs(os.path.join(path, args.text_feature_path), exist_ok=True)

        dataset = AmazonDataset(args, model, path)
        dataloader = AmazonDataloader(args, model).get_dataloader(dataset)

        item_info = []
        for id, vision, vision_path, text, text_path in tqdm(dataloader):
            with torch.no_grad():
                vision_features = model.vision_encode(vision)
                text_features = model.text_encode(text)

            for idx in range(len(id)):
                if vision_path[idx] is not None:
                    torch.save(vision_features[idx, :].clone().detach(), vision_path[idx])
                if text_path[idx] is not None:
                    torch.save(text_features[idx, :].clone().detach(), text_path[idx])

                item_info.append({"id": id[idx],
                                  "vision": os.path.join("./dataset/amazon/preprocess", vision_path[idx]) if vision_path[idx] else None,
                                  "audio": None,
                                  "text": os.path.join("./dataset/amazon/preprocess", text_path[idx]) if text_path[idx] else None})

        with jsonlines.open(os.path.join(path, args.item_out_file), mode="w") as wfile:
            for line in tqdm(item_info, desc="write info"):
                wfile.write(line)


if __name__ == '__main__':
    args = parse_args()
    run(args)






























