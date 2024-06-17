import os
import pickle
import torch
import argparse
import random
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from egcn_base import EDModel as Model
from logger import Log
from loader import load_sentences, update_tag_scheme
from loader import prepare_dataset
from utils import new_masked_cross_entropy, adjust_learning_rate, get_learning_rate
from utils import test_ner, str2bool
from data_utils import iobes_iob, BatchManager, load_word2vec
def showweight(args):
    pos2id_dict = {'PAD' : 0, 'IN': 1, 'DT': 2, 'NNP': 3, 'JJ': 4, 'NNS': 5, ',': 6, 'PRP': 7, 'RB': 8, 'VBD': 9, '.': 10, 'VB': 11, 'CC': 12, 'VBN': 13, 'VBG': 14, 'VBP': 15, 'VBZ': 16, 'CD': 17, 'TO': 18, 'PRP$': 19, 'MD': 20, 'HYPH': 21, ':': 22, 'POS': 23, 'WP': 24, 'WDT': 25, 'RP': 26, 'UH': 27, 'WRB': 28, '``': 29, "''": 30, 'NNPS': 31, 'JJR': 32, '-RRB-': 33, '-LRB-': 34, 'EX': 35, 'JJS': 36, 'RBR': 37, 'NFP': 38, '$': 39, 'RBS': 40, 'PDT': 41, 'FW': 42, 'SYM': 43, 'WP$': 44, 'LS': 45, 'ADD': 46, 'AFX': 47, 'NN': 48}
    pos_category_dict = {"Noun":["NN","NNS","NNP","NNPS"], "Adj":["JJ","JJR","JJS"], "Adv":["RB","RBR","RBS","WRB"], "Verb":["VB","VBD","VBG","VBN","VBP","VBZ"], "Pre_Conj":["IN","RP","CC","TO"], "Pron":["PRP","WP","PRP$","WP$"], "Deter":["DT","PDT","WDT"], "Mood":["MD","UH"], "Other":["PAD","CD","POS","LS","SYM","EX","FW",",", ".", "HYPH", ":", "``", "''", "-RRB-", "-LRB-", "NFP", "$", "ADD", "AFX"]}
    with open(args.map_file, 'rb') as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    old_weights = np.random.rand(len(id_to_char), args.word_embed_dim)
    pre_word_embed = load_word2vec("100.utf8", id_to_char, args.word_embed_dim, old_weights)
    device="cuda"
    e_model = Model(args, id_to_tag, device, pre_word_embed).to(device)
    
    #model_name = "test_dual_unfreezebertclassifier_m075_nocheat_4"
    model_name = "eegcned-master-pos-1.7-elu-wd1e-4"
    print("loading")
    e_model.load_state_dict(torch.load("./models/" + model_name + ".pkl"),strict=False)
    pos_weight_dict = {}
    for name ,param in e_model.named_parameters():
        if(name == "gcn_model.pos_embeddings.weight"):
            pos_matrix = param
            for item in pos2id_dict.items():
                key=item[0]
                value=item[1]
                s=torch.norm(pos_matrix[value],p=1)
                pos_weight_dict[key]=s
            break
        else:
            continue
    print(sorted(pos_weight_dict.items(),key=lambda x:x[1],reverse=True))
    pos_weight_category_dict={}
    
    for item in pos_category_dict.items():
        key=item[0]
        value=item[1]
        weight=0
        for v in value:
            weight=weight+pos_weight_dict[v]
        pos_weight_category_dict[key]=weight/len(value)
    weight_sorted = sorted(pos_weight_category_dict.items(),key=lambda x: x[1], reverse=True) 
    print(weight_sorted)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='egnn for ed')

    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--tag_schema', default="iob", type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_bert', default=0.00001, type=float)
    parser.add_argument('--lr_sharp_decay', default=0, type=int)
    parser.add_argument('--lr_min', default=0.00001, type=float)
    parser.add_argument('--label_weights', default=5, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str, help='Adam;SGD')
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    #parser.add_argument('--num_steps', default=50, type=int)
    parser.add_argument('--num_steps', default=50, type=int)
    parser.add_argument('--word_embed_dim', default=100, type=int)
    parser.add_argument('--bio_embed_dim', default=25, type=int)
    parser.add_argument('--pos_embed_dim', default=25, type=int)
    parser.add_argument('--position_embed_dim', default=0, type=int)
    parser.add_argument('--dep_embed_dim', default=25, type=int)
    parser.add_argument('--rnn_hidden', default=100, type=int)
    parser.add_argument('--rnn_layers', default=1, type=int)
    parser.add_argument('--rnn_dropout', default=0.5, type=float)

    parser.add_argument('--pooling', default='avg', type=str)
    parser.add_argument('--gcn_dim', default=150, type=int)
    parser.add_argument('--num_layers', default=2, type=int, help='num of AGGCN layer blocks')
    parser.add_argument('--gcn_dropout', default=0.5, type=float)
    parser.add_argument('--m', default=0.5, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--map_file', default='maps.pkl', type=str)
    parser.add_argument('--result_path', default='result', type=str)
    parser.add_argument('--emb_file', default='100.utf8', type=str)
    parser.add_argument('--train_file', default=os.path.join("data_doc", "train_ee.json"))
    parser.add_argument('--dev_file', default=os.path.join("data_doc", "dev_ee.json"))
    parser.add_argument('--test_file', default=os.path.join("data_doc", "test_ee.json"))
    parser.add_argument('--log_name', default='test_pos_1.7_bert', type=str)
    parser.add_argument('--seed', default=1023, type=int)
    parser.add_argument('--bert_out_dim', default=100, type=int)
    args = parser.parse_args()
    showweight(args)