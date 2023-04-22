import torch
import wandb
from tqdm import tqdm
from utils.basic_utils import Logger
from metric.retrieval_metric import RetrievalMetric


class Evaluator(object):
    def __init__(self,
                 args,
                 clip_model,
                 fusion_model,
                 eval_dataset,
                 eval_dataloader,
                 **kwargs):
        self.args = args
        self.device = args.device
        self.fusion_embed_dim = args.fusion_embed_dim
        self.logger = Logger(__name__).get_logger(args.log_file)

        self.vision_format = args.eval_vision_format
        self.text_format = args.eval_text_format

        self.clip_model = clip_model
        self.fusion_model = fusion_model
        self.eval_dataset = eval_dataset
        self.eval_dataloader = eval_dataloader
        self.metric = RetrievalMetric()

    def eval(self, epoch):
        self.clip_model.eval()
        self.fusion_model.eval()

        self.logger.info(f"***** Run evaluation *****")
        with torch.no_grad():
            query_dict = {}
            query_mask_dict = {}
            value_dict = {}
            value_mask_dict = {}

            for step, data in tqdm(enumerate(self.eval_dataloader), desc=f"Eval num {self.args.eval_seq_num}"):
                input_ids, vision, vision_mask, text, text_mask = (inputs.to(self.device) for inputs in data)
                vision, vision_mask = self.clip_model(vision, vision_mask, mode="vision", format=self.vision_format)
                text, text_mask = self.clip_model(text, text_mask, mode="text", format=self.text_format)

                keys, query_embeds, query_masks, value_embeds, value_masks = self._eval_step(input_ids, vision, vision_mask, text, text_mask)

                for id, query, q_mask, value, v_mask in zip(keys, query_embeds, query_masks, value_embeds, value_masks):
                    if id not in query_dict:
                        query_dict[id] = []
                    if id not in query_mask_dict:
                        query_mask_dict[id] = []
                    if id not in value_dict:
                        value_dict[id] = []
                    if id not in value_mask_dict:
                        value_mask_dict[id] = []
                    query_dict[id].append(query)
                    query_mask_dict[id].append(q_mask)
                    value_dict[id].append(value)
                    value_mask_dict[id].append(v_mask)

            for key in query_dict.keys():
                q = query_dict[key]
                q_mask = query_mask_dict[key]
                v = value_dict[key]
                v_mask = value_mask_dict[key]

                query_tensor = torch.cat(q, dim=0)
                query_mask = torch.cat(q_mask, dim=0)
                value_tensor = torch.cat(v, dim=0)
                value_mask = torch.cat(v_mask, dim=0)

                R1, R5, R10 = self.metric(query_tensor, query_mask, value_tensor, value_mask)
                if self.args.wandb_enable:
                    wandb.log({f"eval/{key}-R1": R1,
                               f"eval/{key}-R5": R5,
                               f"eval/{key}-R10": R10},
                              commit=False)
                self.logger.info(f"Eval {key} Result: R@1:{R1}, R@5:{R5}, R@10:{R10}")

            if self.args.wandb_enable:
                wandb.log({"train/epoch": epoch})

    def _eval_step(self, input_ids, vision, vision_mask, text, text_mask):
        v_out, v_mask = self.fusion_model(input_ids, vision, vision_mask, text, torch.zeros_like(text_mask))
        v_out = v_out.reshape(-1, self.fusion_embed_dim)
        t_out, t_mask = self.fusion_model(input_ids, vision, torch.zeros_like(vision_mask), text, text_mask)
        t_out = t_out.reshape(-1, self.fusion_embed_dim)

        return ["v2t", "t2v"], [v_out, t_out], [v_mask, t_mask], [t_out, v_out], [t_mask, v_mask]


