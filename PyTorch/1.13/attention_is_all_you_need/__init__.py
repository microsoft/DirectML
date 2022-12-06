from argparse import Namespace
import math
import time
import os
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchbenchmark.util.torchtext_legacy.field import Field
from torchbenchmark.util.torchtext_legacy.data import Dataset
from torchbenchmark.util.torchtext_legacy.iterator import BucketIterator
from torchbenchmark.util.torchtext_legacy.translation import TranslationDataset

from .transformer import Constants
from .transformer.Models import Transformer
from .transformer.Optim import ScheduledOptim
from .train import prepare_dataloaders, cal_performance, patch_src, patch_trg
import random
import numpy as np
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

class Model(BenchmarkModel):
    task = NLP.TRANSLATION
    # Original batch size 256, hardware platform unknown
    # Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/README.md?plain=1#L83
    DEFAULT_TRAIN_BSIZE = 256
    DEFAULT_EVAL_BSIZE = 32
    NUM_OF_BATCHES = 1

    def _create_transformer(self):
        transformer = Transformer(
            self.opt.src_vocab_size,
            self.opt.trg_vocab_size,
            src_pad_idx=self.opt.src_pad_idx,
            trg_pad_idx=self.opt.trg_pad_idx,
            trg_emb_prj_weight_sharing=self.opt.proj_share_weight,
            emb_src_trg_weight_sharing=self.opt.embs_share_weight,
            d_k=self.opt.d_k,
            d_v=self.opt.d_v,
            d_model=self.opt.d_model,
            d_word_vec=self.opt.d_word_vec,
            d_inner=self.opt.d_inner_hid,
            n_layers=self.opt.n_layers,
            n_head=self.opt.n_head,
            dropout=self.opt.dropout).to(self.device)
        
        return transformer

    def _preprocess(self, data_iter):
        preloaded_data = []
        for d in data_iter:
            src_seq = patch_src(d.src, self.opt.src_pad_idx).to(self.device)
            trg_seq, gold = map(lambda x: x.to(self.device), patch_trg(d.trg, self.opt.trg_pad_idx))
            preloaded_data.append((src_seq, trg_seq, gold))
        return preloaded_data

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        root = os.path.join(str(Path(__file__).parent), ".data")
        self.opt = Namespace(**{
            'batch_size': self.batch_size,
            'd_inner_hid': 2048,
            'd_k': 64,
            'd_model': 512,
            'd_word_vec': 512,
            'd_v': 64,
            'data_pkl': f'{root}/m30k_deen_shr.pkl',
            'debug': '',
            'dropout': 0.1,
            'embs_share_weight': False,
            'epoch': 1,
            'label_smoothing': False,
            'log': None,
            'n_head': 8,
            'n_layers': 6,
            'n_warmup_steps': 128,
            'cuda': True,
            'proj_share_weight': False,
            'save_mode': 'best',
            'save_model': None,
            'script': False,
            'train_path': None,
            'val_path': None,
        })

        train_data, test_data = prepare_dataloaders(self.opt, self.device)
        self.model = self._create_transformer()

        if test == "train":
            self.model.train()
            self.example_inputs = self._preprocess(train_data)
            self.optimizer = ScheduledOptim(
                optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                2.0, self.opt.d_model, self.opt.n_warmup_steps)
        elif test == "eval":
            self.model.eval()
            self.example_inputs = self._preprocess(test_data)

    def get_module(self):
        for (src_seq, trg_seq, gold) in self.example_inputs:
            return self.model, (*(src_seq, trg_seq), )

    def eval(self) -> torch.Tensor:
        result = None
        for _, (src_seq, trg_seq, gold) in zip(range(self.NUM_OF_BATCHES), self.example_inputs):
            result = self.model(*(src_seq, trg_seq))
        return (result, )

    def train(self):
        for _, (src_seq, trg_seq, gold) in zip(range(self.NUM_OF_BATCHES), self.example_inputs):
            self.optimizer.zero_grad()
            example_inputs = (src_seq, trg_seq)
            pred = self.model(*example_inputs)
            loss, n_correct, n_word = cal_performance(
                pred, gold, self.opt.trg_pad_idx, smoothing=self.opt.label_smoothing)
            loss.backward()
            self.optimizer.step_and_update_lr()
