import os
import torch
import wandb
import torch.distributed as dist
from model.fusion import FusionEncoder
from configs.config import BasicConfigs
from utils.basic_utils import wandb_init
from utils.basic_utils import seeds_init, Logger
from data.webvid_dataset import WebvidDataset
from data.msrvtt_dataset import MSRVTTDataset
from trainer.pretrain_trainer import Trainer
from model.extractor import ClipEncoder
from loss.combinatorial_loss import CombinatorialLoss
from trainer.pretrain_evaluator import Evaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from data.dataloader import TrainDataLoader, EvalDataLoader


def run(args):
    clip_model = ClipEncoder(args, **args).to(args.device)
    fusion_model = FusionEncoder(args, **args).to(args.device)

    args.vision_resolution = clip_model.model.visual.input_resolution

    train_dataset = WebvidDataset(args,
                                  data_part="train",
                                  item_file=args.train_item_file,
                                  seq_file=args.train_seq_file,
                                  vision_format=args.train_vision_format,
                                  text_format=args.train_text_format,
                                  **args)
    args.train_item_num = train_dataset.get_item_num()
    args.train_seq_num = train_dataset.get_seq_num()
    train_dataloader = TrainDataLoader(args,
                                       dataset=train_dataset,
                                       **args).get_dataloader()

    eval_dataset = MSRVTTDataset(args,
                                 data_part="eval",
                                 item_file=args.eval_item_file,
                                 seq_file=args.eval_seq_file,
                                 vision_format=args.eval_vision_format,
                                 text_format=args.eval_text_format,
                                 **args)
    args.eval_item_num = eval_dataset.get_item_num()
    args.eval_seq_num = eval_dataset.get_seq_num()
    eval_dataloader = EvalDataLoader(args,
                                     dataset=eval_dataset,
                                     **args).get_dataloader()

    evaluator = Evaluator(args,
                          clip_model=clip_model,
                          fusion_model=fusion_model,
                          eval_dataset=eval_dataset,
                          eval_dataloader=eval_dataloader,
                          **args)

    training_parameters = []
    fusion_model = DDP(fusion_model, device_ids=[args.local_rank], output_device=args.local_rank)
    training_parameters.append({"params": fusion_model.parameters(), "lr": args.learning_rate})

    if not len(training_parameters):
        raise Exception("No parameter trainable!")

    optimizer = torch.optim.AdamW(training_parameters)
    optimizer.zero_grad()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scheduler_gamma)
    loss = CombinatorialLoss(args, model=fusion_model, **args)

    trainer = Trainer(args,
                      loss=loss,
                      clip_model=clip_model,
                      fusion_model=fusion_model,
                      train_dataloader=train_dataloader,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      evaluator=evaluator,
                      **args)
    trainer.train()


def main():
    configs = BasicConfigs()
    args = configs.get_training_args()
    seeds_init(args.seed)

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    }
    args.rank = int(env_dict["RANK"])
    args.local_rank = int(env_dict["LOCAL_RANK"])
    args.world_size = int(env_dict["WORLD_SIZE"])
    args.device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else "cpu"

    logger = Logger(__name__).get_logger(args.log_file)
    if args.local_rank == 0:
        if args.wandb_enable:
            wandb_init(args)
        logger.info(args)

    dist.init_process_group(backend="nccl")
    run(args)
    dist.destroy_process_group()
    if args.local_rank == 0 and args.wandb_enable:
        wandb.finish()


if __name__ == "__main__":
    main()
