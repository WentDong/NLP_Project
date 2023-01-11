#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='root of data')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings. If embedding is construct by elmo, it should be equal to 1024.')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')
    #### Algorithm Choose
    arg_parser.add_argument("--algo", default = "Baseline", choices=['Baseline', 'Dual'], help = 'Algorithem to run')
    
    arg_parser.add_argument("--use_bert", default = False, action = "store_true", help = "whether use bert")
    # arg_parser.add_argument("--finetune_bert", default = False, action = "store_true", help = "whehter finetune bert. it only works when use bert.")
    arg_parser.add_argument("--pretrained_model_name", type = str, default = "hfl/chinese-roberta-wwm-ext", help = "which model to use")
    
    arg_parser.add_argument("--use_elmo", default = False, action = "store_true", help = "whether use ELMo")
    arg_parser.add_argument("--elmo_model", type = str, default = "./zhs.model/", help = "The path where the elmo model saved")
    arg_parser.add_argument("--use_dict", default = False, action = "store_true", help = "whether use lexicon as supplements of dictionary with jieba")
    arg_parser.add_argument("--dict_dir_list", type = list, default = ["./data/lexicon/poi_name.txt", "./data/lexicon/ordinal_number.txt", "./data/lexicon/operation_verb.txt"], help = "direction of dicts. It works only when use dict")
    
    arg_parser.add_argument("--use_crf", default = False, action = "store_true", help = "whether use crf.")
    arg_parser.add_argument("--use_focus", default = False, action = "store_true", help = "whether use LSTM-focus as decoder.")

    arg_parser.add_argument("--alpha_filter", default = False, action = "store_true", help = "whether filter out Englist letters in utts")
    arg_parser.add_argument("--rate_head", type = float, default = 0.88, help = "The rate for the head of a word")
    arg_parser.add_argument("--rate_mid", type = float, default = 0.7, help = "The rate for the middle of a word")

    return arg_parser