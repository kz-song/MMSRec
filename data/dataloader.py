import torch
from abc import abstractmethod
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler


class BasicDataloader(object):
    def __init__(self,
                 args,
                 dataset,
                 **kwargs):
        self.args = args

        self.dataset = dataset
        self.num_worker = args.num_worker

    @abstractmethod
    def get_dataloader(self):
        raise NotImplementedError

    def _collect_func(self, data):
        input_ids = []
        vision = []
        vision_mask = []
        text = []
        text_mask = []

        for ids, v, vm, t, tm in data:
            ids = torch.flip(ids, dims=[0])
            input_ids.append(ids)
            v = torch.flip(v, dims=[0])
            vision.append(v)
            vm = torch.flip(vm, dims=[0])
            vision_mask.append(vm)
            t = torch.flip(t, dims=[0])
            text.append(t)
            tm = torch.flip(tm, dims=[0])
            text_mask.append(tm)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_ids = torch.flip(input_ids, dims=[1])
        vision = pad_sequence(vision, batch_first=True, padding_value=0)
        vision = torch.flip(vision, dims=[1])
        vision_mask = pad_sequence(vision_mask, batch_first=True, padding_value=False)
        vision_mask = torch.flip(vision_mask, dims=[1])
        text = pad_sequence(text, batch_first=True, padding_value=0)
        text = torch.flip(text, dims=[1])
        text_mask = pad_sequence(text_mask, batch_first=True, padding_value=False)
        text_mask = torch.flip(text_mask, dims=[1])

        return input_ids, vision, vision_mask, text, text_mask


class TrainDataLoader(BasicDataloader):
    def __init__(self,
                 args,
                 dataset,
                 **kwargs):
        super().__init__(args,
                         dataset=dataset,
                         **kwargs)

        self.batch_size = args.train_batch_size

    def get_dataloader(self):
        distribute_sampler = DistributedSampler(self.dataset, shuffle=True)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_worker,
                                drop_last=False,
                                pin_memory=True,
                                sampler=distribute_sampler,
                                collate_fn=self._collect_func)
        return dataloader


class EvalDataLoader(BasicDataloader):
    def __init__(self,
                 args,
                 dataset,
                 **kwargs):
        super().__init__(args,
                         dataset=dataset,
                         **kwargs)

        self.batch_size = args.eval_batch_size

    def get_dataloader(self):
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_worker,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=self._collect_func)
        return dataloader


class BasicSequentialDataloader(BasicDataloader):
    def __init__(self, args, dataset, **kwargs):
        super().__init__(args, dataset, **kwargs)

    def _collect_func(self, data):
        input_ids = []

        for ids in data:
            ids = torch.flip(ids, dims=[0])
            input_ids.append(ids)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_ids = torch.flip(input_ids, dims=[1])

        return input_ids

    @abstractmethod
    def get_dataloader(self):
        raise NotImplementedError


class TrainSequentialDataloader(BasicSequentialDataloader):
    def __init__(self, args, dataset, **kwargs):
        super().__init__(args, dataset, **kwargs)

        self.batch_size = args.train_batch_size

    def get_dataloader(self):
        distribute_sampler = DistributedSampler(self.dataset, shuffle=True)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_worker,
                                drop_last=False,
                                pin_memory=True,
                                sampler=distribute_sampler,
                                collate_fn=self._collect_func)
        return dataloader


class EvalSequentialDataloader(BasicSequentialDataloader):
    def __init__(self, args, dataset, **kwargs):
        super().__init__(args, dataset, **kwargs)

        self.batch_size = args.eval_batch_size

    def get_dataloader(self):
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_worker,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=self._collect_func)
        return dataloader


class TestSequentialDataloader(BasicSequentialDataloader):
    def __init__(self, args, dataset, **kwargs):
        super().__init__(args, dataset, **kwargs)

        self.batch_size = args.test_batch_size

    def get_dataloader(self):
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_worker,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=self._collect_func)
        return dataloader



