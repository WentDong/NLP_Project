import torch
from torch import nn

class TransformerEmbed(nn.Module):
    def __init__(self, config):
        super(TransformerEmbed, self).__init__()
        from transformers import AutoTokenizer, AutoModel
        self.device = config.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        self.model = AutoModel.from_pretrained(config.pretrained_model_name)
        self.model = self.model.to(config.device)

    def forward(self, utts, max_length):
        inputs = [self.tokenizer(utt, return_tensors="pt").to(self.device) for utt in utts]
        outs = [self.model(**input)['last_hidden_state'][:,1:-1].squeeze(0) for input in inputs]
        # print(outs[0], outs[0].shape)
        outs = [nn.functional.pad(out, pad=(0,0,0,max_length-len(out)), value=0) for out in outs]
        embed = torch.stack(outs).to(self.device)
        return embed

class ELMoEmb():
    def __init__(self, config):
        self.device = config.device
        from elmoformanylangs import Embedder
        self.embed = Embedder(config.elmo_model).sents2elmo
    def __call__(self, input, max_length):
        embs = self.embed(input)
        embs = torch.stack([nn.functional.pad(torch.tensor(emb), pad=(0,0,0,max_length - len(emb)), value = 0) for emb in embs]).to(self.device)
        return embs

class Emb():
    def __init__(self, config):
        self.use_bert = config.use_bert
        self.use_elmo = config.use_elmo
        if not self.use_bert and not self.use_elmo:
            self.embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        else:
            if self.use_bert:
                self.embed = TransformerEmbed(config)
            else:
                self.embed = ELMoEmb(config)

    def __call__(self, input, max_length = 0):
        if self.use_bert or self.use_elmo:
            return self.embed(input, max_length)
        else:
            return self.embed(input)
