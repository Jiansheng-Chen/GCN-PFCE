# encoding = utf-8

import os
import pickle
import torch
import argparse
import random
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from bert import BertClassifier as Model
from logger import Log
from loader import load_sentences, update_tag_scheme
from loader import prepare_dataset
from utils import new_masked_cross_entropy, adjust_learning_rate, get_learning_rate
from utils import test_ner, str2bool
from data_utils import iobes_iob, BatchManager, load_word2vec

from transformers import BertTokenizer
def main():
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_sentences = load_sentences(args.train_file)
    dev_sentences = load_sentences(args.dev_file)
    test_sentences = load_sentences(args.test_file)

    update_tag_scheme(train_sentences, args.tag_schema)
    update_tag_scheme(test_sentences, args.tag_schema)
    update_tag_scheme(dev_sentences, args.tag_schema)
    
    tokenizer = BertTokenizer.from_pretrained("../bert-base-uncased")
    with open(args.map_file, 'rb') as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    train_data = prepare_dataset(
        train_sentences, tokenizer.vocab, tag_to_id, char_to_id
    )
    dev_data = prepare_dataset(
        dev_sentences, tokenizer.vocab, tag_to_id, char_to_id
    )
    test_data = prepare_dataset(
        test_sentences, tokenizer.vocab, tag_to_id, char_to_id
    )
    cls_id=tokenizer.vocab['[CLS]']
    sep_id=tokenizer.vocab['[SEP]']
    pad_id=tokenizer.vocab['[PAD]']
    train_manager = BatchManager(train_data, args.batch_size, args.num_steps, cls_id, sep_id, pad_id)
    dev_manager = BatchManager(dev_data, 100, args.num_steps, cls_id, sep_id, pad_id)
    test_manager = BatchManager(test_data, 100, args.num_steps, cls_id, sep_id, pad_id)
    

    if args.cuda >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #device = torch.device(args.cuda)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device: ", device)

    if args.train:
        train(id_to_char, id_to_tag, train_manager, dev_manager, device)
    f1, res_info = eval_model(id_to_char, id_to_tag, test_manager, device, args.log_name)
    log_handler.info("\n resinfo {} \v F1: {} ".format(res_info, f1))


def eval_model(id_to_char, id_to_tag, test_manager, device, model_name=None):
    print("Eval ......")
    if not model_name:
        model_name = args.log_name
    #old_weights = np.random.rand(len(id_to_char), args.word_embed_dim)
    #pre_word_embed = load_word2vec("100.utf8", id_to_char, args.word_embed_dim, old_weights)
    e_model = Model(args, len(id_to_tag), device).to(device)
    #e_model = Model(args, id_to_tag, device).to(device)
    e_model.load_state_dict(torch.load("./models/" + model_name + ".pkl"))
    print("model loaded ...")

    e_model.eval()
    all_results = []
    for batch in test_manager.iter_batch():

        strs, lens, chars, segs, subtypes, tags, adj, dep, pos, masks_bert, masks_select, chars_ori = batch
        chars = torch.LongTensor(chars).to(device)
        #chars_ori = torch.LongTensor(chars_ori).to(device)
        _lens = torch.LongTensor(lens).to(device)
        #subtypes = torch.LongTensor(subtypes).to(device)
        tags = torch.LongTensor(tags).to(device)
        #adj = torch.FloatTensor(adj).to(device)
        #dep = torch.LongTensor(dep).to(device)
        #pos = torch.LongTensor(pos).to(device)
        masks_bert = torch.LongTensor(masks_bert).to(device)
        masks_select = torch.BoolTensor(masks_select).to(device)
        logits = e_model(chars, _lens, masks_bert, masks_select)

        """ Evaluate """
        # Decode
        batch_paths = []
        for index in range(len(logits)):
            length = lens[index]
            score = logits[index][:length]  # [seq, dim]
            probs = F.softmax(score, dim=-1)  # [seq, dim]
            path = torch.argmax(probs, dim=-1)  # [seq]
            batch_paths.append(path)

        for i in range(len(strs)):
            result = []
            string = strs[i][:lens[i]]
            gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lens[i]]])
            pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lens[i]]])
            for char, gold, pred in zip(string, gold, pred):
                result.append(" ".join([char, gold, pred]))
            all_results.append(result)

    all_eval_lines = test_ner(all_results, args.result_path, args.log_name)
    res_info = all_eval_lines[1].strip()
    f1 = float(res_info.split()[-1])
    print("eval: f1: {}".format(f1))
    return f1, res_info


def train(id_to_char, id_to_tag, train_manager, dev_manager, device):
    old_weights = np.random.rand(len(id_to_char), args.word_embed_dim)#old_weight是一个len(id_to_char)*args.word_embed_dim的矩阵，矩阵中的值是随机初始化的，每一个元素是服从0~1均匀分布的随机样本值，也就是返回的结果中的每一个元素值在0-1之间。
    pre_word_embed = load_word2vec("100.utf8", id_to_char, args.word_embed_dim, old_weights)

    if args.label_weights:#label_weigths应该是用来平衡正负样本不平均的问题，给不是O的标签更大的权重
        label_weights = torch.ones([len(id_to_tag)]) * args.label_weights
        label_weights[0] = 1.0  # none
        label_weights = label_weights.to(device)
    else:
        label_weights = None

    model = Model(args, len(id_to_tag), device).to(device)
    #model = Model(args, id_to_tag, device).to(device)
    optimizer = 0
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("device: ", model.device)
    MAX_F1 = 0

    for epoch in range(args.epoch):

        log_handler.info("Epoch: {} / {} ：".format(epoch + 1, args.epoch))
        log_handler.info("epoch {}, lr: {} ".format(epoch + 1, get_learning_rate(optimizer)))

        loss = train_epoch(model, optimizer, train_manager, label_weights, device)
        log_handler.info("epoch {}, loss : {}".format(epoch + 1, loss))
        f1, dev_model = dev_epoch(epoch, model, dev_manager, id_to_tag, device)
        log_handler.info("epoch {}, f1 : {}".format(epoch + 1, f1))
        if f1 > MAX_F1:
            MAX_F1 = f1
            torch.save(dev_model.state_dict(), "./models/{}.pkl".format(args.log_name))
        log_handler.info("epoch {}, MAX_F1: {}\n".format(epoch + 1, MAX_F1))
        print()


def dev_epoch(epoch, model, dev_manager, id_to_tag, device):
    # dev
    model.eval()
    all_results = []
    for batch in dev_manager.iter_batch():

        strs, lens, chars, segs, subtypes, tags, adj, dep, pos, masks_bert, masks_select, chars_ori= batch
        chars = torch.LongTensor(chars).to(device)
        #chars_ori = torch.LongTensor(chars_ori).to(device)
        _lens = torch.LongTensor(lens).to(device)
        #subtypes = torch.LongTensor(subtypes).to(device)
        tags = torch.LongTensor(tags).to(device)
        #adj = torch.FloatTensor(adj).to(device)
        #dep = torch.LongTensor(dep).to(device)
        #pos = torch.LongTensor(pos).to(device)
        masks_bert = torch.LongTensor(masks_bert).to(device)
        masks_select = torch.BoolTensor(masks_select).to(device)
        #logits,_ = model(chars, _lens, subtypes, adj, dep, pos, masks_bert, masks_select, chars_ori)  # [batch, seq, dim]
        logits = model(chars, _lens, masks_bert, masks_select)
        """ Evaluate """
        # Decode
        batch_paths = []
        for index in range(len(logits)):
            length = lens[index]
            score = logits[index][:length]  # [seq, dim]
            probs = F.softmax(score, dim=-1)  # [seq, dim]
            path = torch.argmax(probs, dim=-1)  # [seq]
            batch_paths.append(path)

        for i in range(len(strs)):
            result = []
            string = strs[i][:lens[i]]
            gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lens[i]]])
            pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lens[i]]])
            for char, gold, pred in zip(string, gold, pred):
                result.append(" ".join([char, gold, pred]))
            all_results.append(result)

    all_eval_lines = test_ner(all_results, args.result_path, args.log_name)
    log_handler.info("epoch: {}, info: {}".format(epoch + 1, all_eval_lines[1].strip()))
    f1 = float(all_eval_lines[1].strip().split()[-1])
    return f1, model


def train_epoch(model, optimizer, train_manager, label_weights, device):
    total_loss = 0
    model.train()
    for i, batch in enumerate(tqdm(train_manager.iter_batch(shuffle=True))):
        optimizer.zero_grad()
        strs, lens, chars, segs, subtypes, tags, adj, dep, pos, masks_bert, masks_select, chars_ori = batch
        chars = torch.LongTensor(chars).to(device)
        #chars_ori = torch.LongTensor(chars_ori).to(device)
        _lens = torch.LongTensor(lens).to(device)
        #subtypes = torch.LongTensor(subtypes).to(device)
        tags = torch.LongTensor(tags).to(device)
        #adj = torch.FloatTensor(adj).to(device)
        #dep = torch.LongTensor(dep).to(device)
        #pos = torch.LongTensor(pos).to(device)
        masks_bert = torch.LongTensor(masks_bert).to(device)
        masks_select = torch.BoolTensor(masks_select).to(device)
        # print('dep: ', dep.shape, dep.sum())

        #logits,_ = model(chars, lens, subtypes, adj, dep, pos, masks_bert,masks_select, chars_ori)
        logits = model(chars, _lens, masks_bert, masks_select)
        loss = new_masked_cross_entropy(logits, tags, _lens, device, label_weights=label_weights)
        total_loss = total_loss + loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / train_manager.num_batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='egnn for ed')

    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--tag_schema', default="iob", type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_sharp_decay', default=0, type=int)
    parser.add_argument('--lr_min', default=1e-5, type=float)
    parser.add_argument('--label_weights', default=5, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str, help='Adam;SGD')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epoch', default=15, type=int)
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
    
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--map_file', default='maps.pkl', type=str)
    parser.add_argument('--result_path', default='result', type=str)
    parser.add_argument('--emb_file', default='100.utf8', type=str)
    parser.add_argument('--train_file', default=os.path.join("data_doc", "train_ee.json"))
    parser.add_argument('--dev_file', default=os.path.join("data_doc", "dev_ee.json"))
    parser.add_argument('--test_file', default=os.path.join("data_doc", "test_ee.json"))
    parser.add_argument('--log_name', default='test_finetune_bert_5_cla', type=str)
    parser.add_argument('--seed', default=1023, type=int)
    parser.add_argument('--bert_out_dim', default=100, type=int)

    args = parser.parse_args()
    log = Log(args.log_name + ".log")
    log_handler = log.getLog()

    log_handler.info("\nArgs: ")
    for arg in vars(args):
        log_handler.info("{}: {}".format(arg, getattr(args, arg)))
    log_handler.info("\n")

    main()
