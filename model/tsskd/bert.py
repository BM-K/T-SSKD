import torch.nn as nn
from transformers import BertModel
from model.tsskd.aux_task import AuxRecoveryMask

class BERTCF(nn.Module):
    def __init__(self, bert, dropout):
        super(BERTCF, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        full_logits, pooling = self.bert(input_ids=inputs['input_ids'],
                                         attention_mask=inputs['attention_mask'],
                                         token_type_ids=inputs['token_type_ids'],
                                         return_dict=False)

        return self.dropout(pooling), full_logits


class BERT(nn.Module):
    def __init__(self, args, tokenizer):
        super(BERT, self).__init__()
        self.args = args

        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BERTCF(bert, self.args.dropout)
        self.projection = nn.Linear(768, 2)
        self.mask_task = AuxRecoveryMask(self.args, d_model=768, vocab_size=len(tokenizer))

    def forward(self, inputs, ss_inputs):
        poolig_logits, full_logits = self.bert(inputs)
        poolig_logits = self.projection(poolig_logits)

        if self.args.self_supervision == 'True':
            mask_logits = self.mask_task(full_logits, ss_inputs)
        else:
            mask_logits = None

        return poolig_logits.view(-1, poolig_logits.size(-1)), mask_logits