#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from model.embed import Emb

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        
        self.embed = Emb(config)

        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)

        if (config.use_crf):
            args_crf = dict({'target_size': config.num_tags, "device": config.device, "average_batch": True})
            from model.CRF import CRF
            self.crf = CRF(**args_crf)
            self.output_layer = nn.Linear(config.hidden_size, config.num_tags+2)
        elif config.use_focus:
            self.index_slices = [2*i+1 for i in range(self.config.num_layer)]  # generated from the reversed path
            self.index_slices = torch.tensor(self.index_slices, dtype=torch.long, device=self.config.device)
            self.output_layer = LSTM_Decoder(config.hidden_size, config.num_tags, config.tag_pad_idx, config.dropout)
        else:
            self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        max_length = max(lengths)
        if (self.config.use_bert or self.config.use_elmo):
            embed = self.embed(batch.utt, max_length)
        else:
            embed = self.embed(input_ids)
        if (self.config.use_bert or self.config.alpha_filter):
            try:
                i = lengths.index(0)
                embed = embed[0:i]
                lengths = lengths[0:i]
                tag_ids = tag_ids[0:i]
                tag_mask = tag_mask[0:i]
            except:
                i = -1 

        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
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
        elif self.config.use_focus:
            enc_h_t, enc_c_t = h_t_c_t
            h_t = torch.index_select(enc_h_t, 0, self.index_slices)
            c_t = torch.index_select(enc_c_t, 0, self.index_slices)
            tag_output = self.output_layer(hiddens, h_t, c_t, tag_mask, tag_ids)
            return tag_output
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


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )

class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, num_tags, pad_id, dropout):
        super(LSTM_Decoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size//2, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.decoder = nn.LSTM(input_size, input_size//2, num_layers=2, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, hiddens, h_t, c_t, mask, labels=None):
        
        output, (dec_h_t, dec_c_t) = self.decoder(hiddens, (h_t, c_t))
        logits = self.output_layer(output)

        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
    # def __init__(self, input_size, num_tags, pad_id, dropout):
    #     super(LSTM_Decoder, self).__init__()
    #     self.num_tags = num_tags
    #     # self.device = device
    #     self.output_layer = nn.Linear(input_size//2, num_tags)
    #     self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
    #     self.decoder = nn.LSTM(num_tags + input_size, input_size//2, num_layers=2, bidirectional=False, batch_first=True, dropout=dropout)

    # def forward(self, hiddens, h_t, c_t, mask, labels=None):
    #     # print(hiddens[:,:,0:hiddens.shape[2]//2].shape)
    #     logits = torch.zeros(hiddens.shape[0], hiddens.shape[1], self.num_tags)
    #     length = hiddens.shape[1]
    #     last_tag = self.output_layer(hiddens[:, 0:1, 0:hiddens.shape[2]//2])
    #     sm_last_tag = torch.softmax(last_tag, dim=-1)
        
    #     for i in range(length):
    #         decode_inputs = torch.cat((hiddens[:, i:i+1], sm_last_tag), 2)
    #         tag_lstm_out, (dec_h_t, dec_c_t) = self.decoder(decode_inputs, (h_t, c_t))
    #         logits[:, i:i+1] = self.output_layer(tag_lstm_out)
    #         # logits = logits.to(self.device)
    #         last_tag = logits[:, i:i+1]
    #         sm_last_tag = torch.softmax(last_tag, dim=-1)
    #         h_t, c_t = dec_h_t, dec_c_t

    #     logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
    #     prob = torch.softmax(logits, dim=-1)
    #     if labels is not None:
    #         loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
    #         return prob, loss
    #     return (prob, )