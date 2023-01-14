import torch
from torch import nn
def decode(model, label_vocab, batch, config):
    batch_size = len(batch)
    labels = batch.labels
    output = model(batch)
    prob = output[0]
    predictions = []
    for i in range(batch_size):
        if i < len(prob):
            if (config.use_crf):
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
        else:
            predictions.append([])
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
    def __init__(self, input_size, num_tags, pad_id, dropout, use_focus, device):
        super(LSTM_Decoder, self).__init__()
        self.num_tags = num_tags
        self.device = device
        self.output_layer = nn.Linear(input_size//2, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.use_focus = use_focus
        if use_focus:
            self.decoder = nn.LSTM(num_tags + input_size, input_size//2, num_layers=2, bidirectional=False, batch_first=True, dropout=dropout)
        else:
            self.decoder = nn.LSTM(input_size, input_size//2, num_layers=2, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, hiddens, h_t, c_t, mask, labels=None):
        if self.use_focus:
            logits = torch.zeros(hiddens.shape[0], hiddens.shape[1], self.num_tags)
            length = hiddens.shape[1]
            last_tag = self.output_layer(hiddens[:, 0:1, 0:hiddens.shape[2]//2])
            sm_last_tag = torch.softmax(last_tag, dim=-1)
            
            for i in range(length):
                decode_inputs = torch.cat((hiddens[:, i:i+1], sm_last_tag), 2)
                tag_lstm_out, (dec_h_t, dec_c_t) = self.decoder(decode_inputs, (h_t, c_t))
                logits[:, i:i+1] = self.output_layer(tag_lstm_out)
                logits = logits.to(self.device)
                last_tag = logits[:, i:i+1]
                sm_last_tag = torch.softmax(last_tag, dim=-1)
                h_t, c_t = dec_h_t, dec_c_t
        else:
            output, (dec_h_t, dec_c_t) = self.decoder(hiddens, (h_t, c_t))
            logits = self.output_layer(output)

        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )