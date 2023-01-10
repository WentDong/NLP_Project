#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx, config.dropout)
        self.num_layers = config.num_layer
        self.device = "cpu"

    def forward(self, batch, train=True):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        # if train == True:
        #     for i in range(input_ids.shape[0]):
        #         for j in range(input_ids.shape[1]-1):
        #             if tag_ids[i][j] == 1 and tag_ids[i][j+1]==1:
        #                 tmp = input_ids[i][j]
        #                 input_ids[i][j] = input_ids[i][j+1]
        #                 input_ids[i][j+1] = tmp
        #                 break

        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)

        enc_h_t, enc_c_t = h_t_c_t
        index_slices = [2*i+1 for i in range(self.num_layers)]  # generated from the reversed path
        index_slices = torch.tensor(index_slices, dtype=torch.long, device=self.device)
        h_t = torch.index_select(enc_h_t, 0, index_slices)
        c_t = torch.index_select(enc_c_t, 0, index_slices)

        # print(hiddens.shape)
        tag_output = self.output_layer(hiddens, h_t, c_t, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch, False)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
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
        
        # print(predictions[0])

        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id, dropout):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size//2, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.decoder = nn.LSTM(num_tags + input_size, input_size//2, num_layers=2, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, hiddens, h_t, c_t, mask, labels=None):
        # print(hiddens[:,:,0:hiddens.shape[2]//2].shape)
        logits = torch.zeros(hiddens.shape[0], hiddens.shape[1], self.num_tags)
        length = hiddens.shape[1]
        last_tag = self.output_layer(hiddens[:, 0:1, 0:hiddens.shape[2]//2])
        sm_last_tag = torch.softmax(last_tag, dim=-1)

        for i in range(length):
            decode_inputs = torch.cat((hiddens[:, i:i+1], sm_last_tag), 2)
            # print(h_t.shape)
            tag_lstm_out, (dec_h_t, dec_c_t) = self.decoder(decode_inputs, (h_t, c_t))
            # print(tag_lstm_out.shape)
            logits[:, i:i+1] = self.output_layer(tag_lstm_out)
            last_tag = logits[:, i:i+1]
            sm_last_tag = torch.softmax(last_tag, dim=-1)
            h_t, c_t = dec_h_t, dec_c_t

        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
