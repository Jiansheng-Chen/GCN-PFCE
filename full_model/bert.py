import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, args, classes_num, device):
        super(BertClassifier, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained('../bert-base-uncased/')
        self.bert_clafc = nn.Linear(args.bert_dim,classes_num)
        
        
    def forward(self, token_ids, x_len, masks_bert, masks_select):
        
        BATCH_SIZE = token_ids.shape[0]
        BATCH_MAX_LEN = x_len[0]
        token_ids = token_ids[: , :BATCH_MAX_LEN+2].contiguous()
        masks_bert = masks_bert[: , :BATCH_MAX_LEN+2].contiguous()
        masks_select = masks_select[: , :BATCH_MAX_LEN+2].contiguous()
        word_emb = self.bert(input_ids=token_ids, attention_mask=masks_bert)[0]
        x_emb = torch.masked_select(word_emb,masks_select.unsqueeze(-1)).view(BATCH_SIZE,BATCH_MAX_LEN,-1)
        logit = self.bert_clafc(x_emb)
        return logit