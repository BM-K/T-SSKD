import logging
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim
from model.loss import LossF
from model.utils import Metric
from model.tsskd.bert import BERT
from data.dataloader import get_loader
from model.tsskd.transformer import Transformer
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.lossf = LossF(args)
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': -1, 'iter': -1, 'acc': -1}

    def run(self, inputs, ss_inputs, labels):
        clf_logits, ss_logits = self.config['model'](inputs, ss_inputs)

        if self.args.have_teacher == 'True':
            if self.args.self_supervision == 'True':
                clf_distil_loss = self.lossf.distilled(self.config, clf_logits, inputs, labels)
                ss_distil_loss = self.lossf.distilled(self.config, ss_inputs, ss_logits)
            else:
                clf_distil_loss = self.lossf.distilled(self.config, clf_logits, inputs, labels)
        else:
            loss = self.lossf.base(self.config, clf_logits, labels)

        with torch.no_grad():
            acc = self.metric.cal_acc(clf_logits, labels)

        return loss, acc

    def progress(self, loss, acc):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1
        self.model_progress['acc'] += acc

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']
        acc = self.model_progress['acc'].data.cpu().numpy() / self.model_progress['iter']

        return loss, acc

    def get_object(self, tokenizer, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(optim,
                    num_warmup_steps=self.args.warmup_ratio*train_total,
                    num_training_steps=train_total)

        return scheduler

    def model_setting(self):
        loader, tokenizer = get_loader(self.args, self.metric)

        if self.args.have_teacher == 'True':
            model = Transformer(self.args, tokenizer)
            teacher = BERT(self.args, tokenizer)
            model.to(self.args.device)
            teacher.to(self.args.device)
        else:
            if self.args.base_model == 'BERT':
                model = BERT(self.args, tokenizer)
            else:
                model = Transformer(self.args, tokenizer)

            teacher = None
            model.to(self.args.device)

        criterion, optimizer = self.get_object(tokenizer, model)
        scheduler = self.get_scheduler(optimizer, loader['train'])

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'args': self.args,
                  'model': model,
                  'teacher': teacher}

        config = self.metric.move2device(config, self.args.device)

        if config['args'].have_teacher == 'True':
            config['teacher'].load_state_dict(torch.load(config['args'].saved_teacher_model))
            for p in config['teacher'].parameters():
                p.requires_grad_(False)

        self.config = config

        return self.config

    def train(self):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):
            self.config['optimizer'].zero_grad()

            inputs, labels, aux = batch
            loss, acc = self.run(inputs, aux, labels)

            loss.backward()

            self.config['optimizer'].step()
            self.config['scheduler'].step()
            self.progress(loss, acc)

        return self.return_value()

    def valid(self):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):

                inputs, labels, aux = batch
                loss, acc = self.run(inputs, aux, labels)

                self.progress(loss, acc)

        return self.return_value()

    def test(self):

        self.config['model'].load_state_dict(torch.load(self.args.path_to_saved_model))
        self.config['model'].eval()

        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):

                inputs, labels, aux = batch
                loss, acc = self.run(inputs, aux, labels)

                self.progress(loss, acc)

        return self.return_value()