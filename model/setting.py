import torch
import random
import logging
import numpy as np
from argparse import ArgumentParser


class Arguments():

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self):
        self.add_argument('--base_model', type=str, default='BERT')
        self.add_argument('--train', type=str, default='True')
        self.add_argument('--test', type=str, default='True')
        self.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def add_hyper_parameters(self):
        self.add_argument('--patient', type=int, default=5)
        self.add_argument('--dropout', type=int, default=0.1)
        self.add_argument('--max_len', type=int, default=128)
        self.add_argument('--batch_size', type=int, default=32)
        self.add_argument('--epochs', type=int, default=100)
        self.add_argument('--seed', type=int, default=1234)
        self.add_argument('--lr', type=float, default=0.00001)
        self.add_argument('--warmup_ratio', type=float, default=0.1)

    def add_data_parameters(self):
        self.add_argument('--train_data', type=str, default='train.tsv')
        self.add_argument('--test_data', type=str, default='test.tsv')
        self.add_argument('--valid_data', type=str, default='valid.tsv')
        self.add_argument('--task', type=str, default='IMDB')
        self.add_argument('--path_to_data', type=str, default='./data/')
        self.add_argument('--model_ckpt', type=str, default='./checkpoints/.pt')
        self.add_argument('--teacher_ckpt', type=str, default='./checkpoints/.pt')

    def add_distillation_parameters(self):
        self.add_argument('--have_teacher', type=str, default='False')
        self.add_argument('--self_supervision', type=str, default='False')
        self.add_argument('--alpha', type=float, default=0.1)
        self.add_argument('--temperature', type=int, default=3)

    def add_transformer_parameters(self):
        # IMDB
        self.add_argument('--d_model', type=int, default=64)
        self.add_argument('--feedforward', type=int, default=128)
        self.add_argument('--n_heads', type=int, default=8)
        self.add_argument('--n_layers', type=int, default=6)

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:print("argparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:print("\t", key, ":", value, "\n}")
            else:print("\t", key, ":", value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        self.print_args(args)

        return args


class Setting():

    def set_logger(self):

        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        _logger.addHandler(stream_handler)
        _logger.setLevel(logging.DEBUG)

        return _logger

    def set_seed(self, args):

        seed = args.seed

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run(self):

        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_hyper_parameters()
        parser.add_data_parameters()
        parser.add_distillation_parameters()

        args = parser.parse()
        logger = self.set_logger()
        self.set_seed(args)

        return args, logger