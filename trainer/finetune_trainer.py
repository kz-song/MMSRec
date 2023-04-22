import torch
import wandb
from tqdm import tqdm
import torch.distributed as dist
from utils.basic_utils import Logger, check_dirs
from loss.ce_loss import CELoss


class Trainer(object):
    def __init__(self,
                 args,
                 evaluator,
                 optimizer,
                 fusion_model,
                 lr_scheduler,
                 train_dataset,
                 train_dataloader,
                 **kwargs):

        self.args = args
        self.device = args.device
        self.model_save_path = args.model_save_path
        self.model_resume_path = args.model_resume_path
        self.logger = Logger(__name__).get_logger(args.log_file)

        self.vision_format = args.train_vision_format
        self.text_format = args.train_text_format

        self.fusion_model = fusion_model

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator

        self.ce_loss = CELoss(args, **args)

        if self.model_resume_path is not None:
            self._resume_checkpoint(self.model_resume_path)

    def _save_checkpoint(self, epoch):
        output_path = self.model_save_path
        check_dirs(output_path)
        self.logger.info(f"***** Saving model checkpoint {epoch} to {output_path}")

        self.fusion_model.module.save_pretrained(output_path)

    def _resume_checkpoint(self, path):
        if self.args.local_rank == 0:
            self.logger.info(f"***** Resume Model from path {path}")

        map_location = {"cuda:0": f"cuda:{self.args.local_rank}"}
        self.fusion_model.module.from_pretrained(path, map_location)

    def train(self):
        if self.args.local_rank == 0:
            self.logger.info("***** Run training *****")
            self.logger.info(f"  Num examples = {self.args.train_seq_num}")
            self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
            self.logger.info(f"  Total train batch size = {self.args.world_size * self.args.train_batch_size}")
            self.logger.info(f"  Total steps = {self.args.num_train_epochs * len(self.train_dataloader)}")

        early_stopping_count = torch.tensor(0, dtype=torch.int, device=self.args.device)
        eval_max = -1

        for epoch in range(self.args.num_train_epochs):
            self._train_epoch(epoch)

            if self.args.local_rank == 0:
                torch.cuda.empty_cache()
                eval_result = self.evaluator.eval(epoch)
                early_stopping_count += 1

                if eval_result > eval_max:
                    eval_max = eval_result
                    early_stopping_count = torch.tensor(0, dtype=torch.int, device=self.args.device)
                    self._save_checkpoint(epoch)

            dist.broadcast(early_stopping_count, src=0)
            if early_stopping_count >= self.args.early_stopping:
                break

        if self.args.local_rank == 0:
            torch.cuda.empty_cache()
            self._resume_checkpoint(self.model_save_path)
            self.evaluator.test()

    def _train_epoch(self, epoch):
        self.fusion_model.train()
        self.train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(self.train_dataloader)), disable=(self.args.local_rank != 0))

        for step, data in enumerate(self.train_dataloader):
            progress_bar.set_description(f"Training epoch {epoch}, steps {step} ")

            input_ids = data.to(self.device)

            pred, label = self.fusion_model(input_ids, mode="mlm")
            total_loss = self.ce_loss(pred, label)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            progress_bar.update(1)
            if self.args.local_rank == 0 and self.args.wandb_enable:
                wandb.log({"train/total_loss": total_loss.item(),
                           "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                           "train/epoch": epoch},
                          step=epoch * len(self.train_dataloader) + step)

        self.lr_scheduler.step()









