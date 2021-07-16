import copy
import torch
import logging
from random import *
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer):
        self.args = args
        self.metric = metric
        self.max_mask_size = args.max_len

        self.label = []
        self.sentence = []
        self.token_type_ids = []
        self.attention_mask = []

        self.masked_source = []
        self.masked_tokens = []
        self.masked_position = []

        self.bert_tokenizer = tokenizer
        self.vocab_size = len(self.bert_tokenizer)
        self.file_path = file_path

        """
        init token, idx = [CLS], 101
        pad token, idx = [PAD], 0
        unk token, idx = [UNK], 100
        eos token, idx = [EOS], 30522
        max token, idx = [MASK], 103
        """

        self.bert_tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        self.init_token = self.bert_tokenizer.cls_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.eos_token = self.bert_tokenizer.eos_token
        self.mask_token = self.bert_tokenizer.mask_token

        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.eos_token)
        self.mask_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.mask_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self, type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                sentence, token_type, attention_mask, label = self.data2tensor(line)

                tensored_src_for_mask = copy.deepcopy(sentence)

                masked_source, masked_tokens, masked_position = \
                    self.get_masked_source(tensored_src_for_mask)
                self.masked_source.append(masked_source)
                self.masked_tokens.append(masked_tokens)
                self.masked_position.append(masked_position)

                self.sentence.append(sentence)
                self.token_type_ids.append(token_type)
                self.attention_mask.append(attention_mask)
                self.label.append(label)

        assert len(self.sentence) == \
               len(self.label) == \
               len(self.masked_source) == \
               len(self.masked_tokens) == \
               len(self.masked_position)

    """
    Converting text data to tensor &
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        split_data = line.split('\t')
        sentence, label = split_data[0], split_data[1]

        sentence_tokens = self.bert_tokenizer(sentence, return_tensors="pt",
                                              max_length=self.args.max_len,
                                              pad_to_max_length="right")

        input_ids = sentence_tokens['input_ids']
        token_type_ids = sentence_tokens['token_type_ids']
        attention_mask = sentence_tokens['attention_mask']

        return torch.tensor(input_ids).squeeze(0),\
               torch.tensor(token_type_ids).squeeze(0),\
               torch.tensor(attention_mask).squeeze(0),\
               torch.tensor(int(label))

    # Making masked data for Aux task mini-batch
    # Reference : https://github.com/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.py
    def get_masked_source(self, source):

        ori_src = copy.deepcopy(source)

        try:
            start_padding_idx = (source == self.pad_token_idx).nonzero()[0].data.cpu().numpy()[0]
        except IndexError:
            start_padding_idx = self.args.max_len

        source = source[:start_padding_idx]

        n_pred = min(self.max_mask_size, max(1, int(round(len(source) * 0.15))))  # mask 15%
        cand_maked_pos = [i for i, token in enumerate(source)]

        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(ori_src[pos])
            if random() < 0.8:  # 80%
                source[pos] = self.mask_token_idx
            elif random() < 0.5:  # 10%
                index = randint(0, self.vocab_size - 1)
                source[pos] = index

        masked_source = list(copy.deepcopy(source.data.numpy()))
        for i in range(self.args.max_len - len(source)): masked_source.append(self.pad_token_idx)

        # Zero Padding (100% - 15%) tokens
        if self.max_mask_size > n_pred:
            n_pad = self.max_mask_size - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        return torch.tensor(masked_source), torch.tensor(masked_tokens), torch.tensor(masked_pos)

    def __getitem__(self, index):

        auxiliary_data = {'mask': {
            'source': self.masked_source[index],
            'tokens': self.masked_tokens[index],
            'position': self.masked_position[index]}}

        input_data = {'input_ids': self.sentence[index],
                      'token_type_ids': self.token_type_ids[index],
                      'attention_mask': self.attention_mask[index]}

        input_data =self.metric.move2device(input_data, self.args.device)
        auxiliary_data = self.metric.move2device(auxiliary_data, self.args.device)

        return input_data, self.label[index].to(self.args.device), auxiliary_data

    def __len__(self):
        return len(self.label)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    path_to_train_data = args.path_to_data+'/'+args.task+'/'+args.train_data
    path_to_valid_data = args.path_to_data+'/'+args.task+'/'+args.valid_data
    path_to_test_data = args.path_to_data+'/'+args.task+'/'+args.test_data

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer)
    valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer)
    test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer)

    train_iter.load_data('train')
    valid_iter.load_data('valid')
    test_iter.load_data('test')

    loader = {'train': DataLoader(dataset=train_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'valid': DataLoader(dataset=valid_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'test': DataLoader(dataset=test_iter,
                                 batch_size=args.batch_size,
                                 shuffle=True)}

    return loader, tokenizer


if __name__ == '__main__':
    get_loader('test')