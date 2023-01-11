#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from model.embed import Emb
from model.Decoder import decode
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
        elif config.use_lstm_decoder:
            self.index_slices = [2*i+1 for i in range(self.config.num_layer)]  # generated from the reversed path
            self.index_slices = torch.tensor(self.index_slices, dtype=torch.long, device=self.config.device)
            from model.Decoder import LSTM_Decoder
            self.output_layer = LSTM_Decoder(config.hidden_size, config.num_tags, config.tag_pad_idx, config.dropout, config.use_focus, config.device)
        else:
            from model.Decoder import TaggingFNNDecoder
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
        elif self.config.use_lstm_decoder:
            enc_h_t, enc_c_t = h_t_c_t
            h_t = torch.index_select(enc_h_t, 0, self.index_slices)
            c_t = torch.index_select(enc_c_t, 0, self.index_slices)
            tag_output = self.output_layer(hiddens, h_t, c_t, tag_mask, tag_ids)
            return tag_output
        else:
            tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
            return tag_output

    def decode(self, label_vocab, batch):
        return decode(self, label_vocab, batch, self.config)

