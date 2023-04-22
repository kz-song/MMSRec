import torch
import wandb
from tqdm import tqdm
from utils.basic_utils import Logger
from metric.predict_metric import PredictMetric


def _calc_Recall(sort_lists, batch_size, topk=10):
    Recall_result = torch.sum(sort_lists < topk) / batch_size
    return Recall_result


def _calc_NDCG(sort_lists, batch_size, topk=10):
    hit = sort_lists < topk
    NDCG_score = hit * (1 / torch.log2(sort_lists + 2))
    NDCG_result = torch.sum(NDCG_score) / batch_size
    return NDCG_result


class Evaluator(object):
    def __init__(self,
                 args,
                 fusion_model,
                 eval_dataset,
                 eval_dataloader,
                 test_dataset,
                 test_dataloader,
                 **kwargs):
        self.args = args
        self.device = args.device
        self.logger = Logger(__name__).get_logger(args.log_file)

        self.fusion_model = fusion_model

        self.eval_dataset = eval_dataset
        self.eval_dataloader = eval_dataloader
        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader
        self.metric = PredictMetric()

    @torch.no_grad()
    def eval(self, epoch):
        self.fusion_model.eval()

        self.logger.info(f"***** Run eval *****")
        sort_lists = []
        batch_size = 0

        for step, data in tqdm(enumerate(self.eval_dataloader)):
            input_ids = data.to(self.device)
            pred, label = self.fusion_model(input_ids, mode="pred")
            sort_index, batch = self.metric(pred, label)

            sort_lists.append(sort_index)
            batch_size += batch

        sort_lists = torch.cat(sort_lists, dim=0)

        Recall10 = _calc_Recall(sort_lists, batch_size, 10)
        Recall50 = _calc_Recall(sort_lists, batch_size, 50)
        NDCG10 = _calc_NDCG(sort_lists, batch_size, 10)
        NDCG50 = _calc_NDCG(sort_lists, batch_size, 50)

        if self.args.wandb_enable:
            wandb.log({"eval/Recall@10": Recall10,
                       "eval/Recall@50": Recall50,
                       "eval/NDCG@10": NDCG10,
                       "eval/NDCG@50": NDCG50,
                       "train/epoch": epoch})
        self.logger.info(f"Epoch {epoch} Eval Result: R@10:{Recall10}, R@50:{Recall50}, NDCG@10:{NDCG10}, NDCG@50:{NDCG50}")

        return Recall10

    @torch.no_grad()
    def test(self):
        self.fusion_model.eval()

        self.logger.info(f"***** Run test *****")
        sort_lists = []
        batch_size = 0

        for step, data in tqdm(enumerate(self.test_dataloader)):
            input_ids = data.to(self.device)
            pred, label = self.fusion_model(input_ids, mode="pred")
            sort_index, batch = self.metric(pred, label)

            sort_lists.append(sort_index)
            batch_size += batch

        sort_lists = torch.cat(sort_lists, dim=0)

        Recall10 = _calc_Recall(sort_lists, batch_size, 10)
        Recall50 = _calc_Recall(sort_lists, batch_size, 50)
        NDCG10 = _calc_NDCG(sort_lists, batch_size, 10)
        NDCG50 = _calc_NDCG(sort_lists, batch_size, 50)

        if self.args.wandb_enable:
            wandb.log({"test/Recall@10": Recall10,
                       "test/Recall@50": Recall50,
                       "test/NDCG@10": NDCG10,
                       "test/NDCG@50": NDCG50})
        self.logger.info(f"Test Result: R@10:{Recall10}, R@50:{Recall50}, NDCG@10:{NDCG10}, NDCG@50:{NDCG50}")



