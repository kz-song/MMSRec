import os
import wandb
import torch
import torch.distributed as dist
from model.fusion import FusionEncoderForMaskedLM
from configs.config import BasicConfigs
from utils.basic_utils import wandb_init
from trainer.finetune_evaluator import Evaluator
from utils.basic_utils import seeds_init, Logger
from data.movielens_dataset import MovieLensSequentialDataset
from model.extractor import ClipEncoder
from trainer.finetune_trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from data.dataloader import TrainSequentialDataloader, EvalSequentialDataloader, TestSequentialDataloader


def run(args):
    clip_model = ClipEncoder(args, **args).to(args.device).eval()
    args.vision_resolution = clip_model.model.visual.input_resolution

    train_dataset = MovieLensSequentialDataset(args,
                                               data_part="train",
                                               item_file=args.train_item_file,
                                               seq_file=args.train_seq_file,
                                               vision_format=args.train_vision_format,
                                               text_format=args.train_text_format,
                                               **args)
    args.train_item_num = train_dataset.get_item_num()
    args.train_seq_num = train_dataset.get_seq_num()
    train_dataloader = TrainSequentialDataloader(args,
                                                 dataset=train_dataset,
                                                 **args).get_dataloader()

    eval_dataset = MovieLensSequentialDataset(args,
                                              data_part="eval",
                                              item_file=args.eval_item_file,
                                              seq_file=args.eval_seq_file,
                                              vision_format=args.eval_vision_format,
                                              text_format=args.eval_text_format,
                                              **args)
    args.eval_item_num = eval_dataset.get_item_num()
    args.eval_seq_num = eval_dataset.get_seq_num()
    eval_dataloader = EvalSequentialDataloader(args,
                                               dataset=eval_dataset,
                                               **args).get_dataloader()

    test_dataset = MovieLensSequentialDataset(args,
                                              data_part="test",
                                              item_file=args.test_item_file,
                                              seq_file=args.test_seq_file,
                                              vision_format=args.test_vision_format,
                                              text_format=args.test_text_format,
                                              **args)
    args.test_item_num = test_dataset.get_item_num()
    args.test_seq_num = test_dataset.get_seq_num()
    test_dataloader = TestSequentialDataloader(args,
                                               dataset=test_dataset,
                                               **args).get_dataloader()

    fusion_model = FusionEncoderForMaskedLM(args, **args).to(args.device)
    fusion_model.modality_embedding_init(train_dataset, clip_model)
    del clip_model

    evaluator = Evaluator(args,
                          fusion_model=fusion_model,
                          eval_dataset=eval_dataset,
                          eval_dataloader=eval_dataloader,
                          test_dataset=test_dataset,
                          test_dataloader=test_dataloader,
                          **args)

    training_parameters = []
    fusion_model = DDP(fusion_model, device_ids=[args.local_rank], output_device=args.local_rank)
    training_parameters.append({"params": fusion_model.parameters(), "lr": args.learning_rate})

    if not len(training_parameters):
        raise Exception("No parameter trainable!")

    optimizer = torch.optim.AdamW(training_parameters)
    optimizer.zero_grad()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scheduler_gamma)

    trainer = Trainer(args,
                      fusion_model=fusion_model,
                      train_dataset=train_dataset,
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
    if args.wandb_enable:
        wandb.finish()


if __name__ == "__main__":
    main()























