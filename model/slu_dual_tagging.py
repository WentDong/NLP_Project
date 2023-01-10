#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from model.slu_baseline_tagging import TaggingFNNDecoder
# from model.Word_Adaper import Word_Adapter
import jieba

from model.embed import ELMoEmb
def Feat_Merge(out_word, Batch_splits, out_char, device, rate_head = 0.88, rate_mid = 0.7):
    feat_B = torch.zeros_like(out_char).to(device)
    for i,split in enumerate(Batch_splits):
        s = 0
        for j,len in enumerate(split):
            # feat_B[i][s:s+len] = out_feat[i][j].unsqueeze(0).repeat((len,1))
            feat_B[i][s] = out_word[i][j].unsqueeze(0)*rate_head + out_char[i][s].unsqueeze(0) * (1-rate_head)
            if (len>1):
                feat_B[i][s+1:s+len] = out_word[i][j].unsqueeze(0).repeat((len-1,1))*rate_mid + out_char[i][s+1:s+len].unsqueeze(0)*(1-rate_mid)
            s += len
    return feat_B

class SLUTagging(nn.Module):
    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell

        self.embed = ELMoEmb(config)
        
        self.rnn_char = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.rnn_word = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        # self.adapter = Word_Adapter(input_dim=config.hidden_size) Don't Work!
        self.dropout_layer = nn.Dropout(p=config.dropout)
        
        if config.use_dict:
            for x in config.dict_dir_list:
                jieba.load_userdict(x)
                
        if (config.use_crf):
            args_crf = dict({'target_size': config.num_tags, "device": config.device, "average_batch": True})
            from model.CRF import CRF
            self.crf = CRF(**args_crf)
            self.output_layer = nn.Linear(config.hidden_size, config.num_tags+2)
        else:
            self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input = batch.utt # B x str
        lengths_char = batch.lengths # B x len
        seglist = [jieba.lcut(str, cut_all=False) for str in input]
        Batch_split = []
        lengths_word = []
        for seg in seglist:
            split = []
            for str in seg:
                split.append(len(str))
            Batch_split.append(split)
            lengths_word.append(len(seg))

        mx_char = max(lengths_char)
        char_emb = self.embed(input, mx_char)

        mx_word = max(lengths_word)
        word_emb = self.embed(seglist, mx_word)

        packed_inputs_char = rnn_utils.pack_padded_sequence(char_emb, lengths_char, batch_first=True, enforce_sorted=False)
        packed_inputs_word = rnn_utils.pack_padded_sequence(word_emb, lengths_word, batch_first=True, enforce_sorted=False)

        packed_rnn_outs_char, _ = self.rnn_char(packed_inputs_char) # B x Len x Dim
        packed_rnn_outs_word, _ = self.rnn_word(packed_inputs_word)

        rnn_out_char, _ = rnn_utils.pad_packed_sequence(packed_rnn_outs_char, batch_first=True)
        rnn_out_word, _ = rnn_utils.pad_packed_sequence(packed_rnn_outs_word, batch_first=True)

        rnn_out = Feat_Merge(rnn_out_word, Batch_split, rnn_out_char, self.config.device, self.config.rate_head, self.config.rate_mid)
        
        # rnn_out = self.adapter(rnn_out_char, rnn_out_word_repeat) # Not Work
        
        hiddens = self.dropout_layer(rnn_out)

        if self.config.use_crf:
            tag_output = self.output_layer(hiddens)
            if tag_ids == None:
                _, best_path = self.crf(tag_output, tag_mask)
                return (best_path,)
            else:
                _, best_path = self.crf(tag_output, tag_mask)
                loss = self.crf.neg_log_likelihood_loss(tag_output, tag_mask.bool(), tag_ids)
                return (best_path, loss)
        else:
            tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
            return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            if (self.config.use_crf):
                pred = prob[i].cpu().tolist()
                pred_tuple = []
                idx_buff, tag_buff, pred_tags = [], [], []
                pred = pred[:len(batch.utt[i])]
            else:
                pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
                pred_tuple = []
                idx_buff, tag_buff, pred_tags = [], [], []
                pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()
