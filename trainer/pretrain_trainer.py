import os
import torch
import wandb
from tqdm import tqdm
from utils.basic_utils import Logger, check_dirs


class Trainer(object):
    def __init__(self,
                 args,
                 loss,
                 evaluator,
                 optimizer,
                 clip_model,
                 fusion_model,
                 lr_scheduler,
                 train_dataloader,
                 **kwargs):

        self.args = args
        self.device = args.device
        self.model_save_path = args.model_save_path
        self.model_resume_checkpoint = args.model_resume_checkpoint
        self.logger = Logger(__name__).get_logger(args.log_file)

        self.vision_format = args.train_vision_format
        self.text_format = args.train_text_format

        self.clip_model = clip_model
        self.fusion_model = fusion_model

        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.loss = loss

        if self.model_resume_checkpoint is not None:
            self._resume_checkpoint(self.model_resume_checkpoint)

    def _save_checkpoint(self, epoch):
        checkpoint_folder = f"checkpoint-epoch-{epoch}"
        output_path = os.path.join(self.model_save_path, checkpoint_folder)
        check_dirs(output_path)
        self.logger.info(f"***** Saving model checkpoint to {output_path}")

        self.fusion_model.module.save_pretrained(output_path)

    def _resume_checkpoint(self, path):
        if self.args.local_rank == 0:
            self.logger.info(f"***** Resume Trainer from checkpoint {path}")

        map_location = {"cuda:0": f"cuda:{self.args.local_rank}"}

        self.fusion_model.module.from_pretrained(path, map_location)

    def _set_model_state(self):
        self.clip_model.eval()
        self.fusion_model.train()

    def train(self):
        if self.args.local_rank == 0:
            self.logger.info("***** Run training *****")
            self.logger.info(f"  Num examples = {self.args.train_seq_num}")
            self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
            self.logger.info(f"  Total train batch size = {self.args.world_size * self.args.train_batch_size}")
            self.logger.info(f"  Total steps = {self.args.num_train_epochs * len(self.train_dataloader)}")

        self.progress_bar = tqdm(range(self.args.num_train_epochs * len(self.train_dataloader)), disable=(self.args.local_rank != 0))
        for epoch in range(self.args.num_train_epochs):
            self._train_epoch(epoch)

            if self.args.local_rank == 0:
                torch.cuda.empty_cache()
                self.evaluator.eval(epoch)

            if self.args.local_rank == 0 and (epoch + 1) % self.args.save_epochs == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self._set_model_state()
        self.train_dataloader.sampler.set_epoch(epoch)

        for step, data in enumerate(self.train_dataloader):
            self.progress_bar.set_description(f"Training epoch {epoch}, steps {step} ")

            input_ids, vision, vision_mask, text, text_mask = (inputs.to(self.device) for inputs in data)

            with torch.no_grad():
                vision, vision_mask = self.clip_model(vision, vision_mask, mode="vision", format=self.vision_format)
                text, text_mask = self.clip_model(text, text_mask, mode="text", format=self.text_format)

            combinatorial_loss = self.loss(input_ids, vision, vision_mask, text, text_mask)

            total_loss = combinatorial_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.progress_bar.update(1)
            if self.args.local_rank == 0 and self.args.wandb_enable:
                wandb.log({"train/combinatorial_loss": combinatorial_loss.item(),
                           "train/total_loss": total_loss.item(),
                           "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                           "train/epoch": epoch},
                          step=epoch * len(self.train_dataloader) + step)

        self.lr_scheduler.step()


