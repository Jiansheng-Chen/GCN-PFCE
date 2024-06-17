# encoding = utf8
import re, math, codecs, random
import numpy as np
from tqdm import trange

def iob2(tags):#判断标签是否是BIO格式，是就返回True，不是就返回False
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
    return True

def iob_iobes(tags):#这个函数把bio格式变成bioes格式，其实就是如果b开头的但是后面没有i的话，就把b变成s，意味着单个字的触发词；如果i开头，但是是结尾的话，就把i变成e，意味着这是触发词结尾的地方。
    """
    IOB -> IOBES
    """
    new_tags = []
    for i , tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-','S-'))
        elif tag.split('-')[0] == 'I':
            if i+1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def create_dico(item_list):

    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):

    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i,v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def create_input(data):

    inputs = list()
    inputs.append(data['chars'])
    # inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embedding.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with'
          'pretrained embeddings.'% (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights

def get_doc_features(doc_id, char_to_id, doc_dict, chars):
    sentence_num = 8
    doc_sentence = doc_dict[doc_id[0]] 
    doc_chars = list()
    for sentence in doc_sentence:
        doc_char = [char_to_id[w if w in char_to_id else '<UNK>'] for w in sentence]
        doc_chars.append(doc_char)  
    a = doc_chars.index(chars) 
    if len(doc_chars) <= sentence_num:
        doc_chars = doc_chars
    else: #
        if a <= sentence_num/2:
            doc_chars = doc_chars[:sentence_num]
        elif len(doc_chars)-a <= sentence_num/4:
            doc_chars = doc_chars[-sentence_num:0]
        else:
            doc_chars = doc_chars[int(a-sentence_num/2):int(a+sentence_num/2)]
    return doc_chars


def get_dep_features(string, dep_rels):
    #这个函数把dependency label映射成id返回
    dep_dict = { 'O': 0, 'punct': 1, 'iobj': 2, 'parataxis': 3, 'auxpass': 4, 'aux': 5, 'conj': 6, 'advcl': 7, 'acl:relcl': 8, 'nsubjpass': 9,'csubj': 10, 'compound': 11, 'compound:prt': 12, 'mwe': 13, 'cop': 14, 'neg': 15, 'nmod:poss': 16, 'appos': 17, 'cc:preconj': 18, 'nmod': 19, 'nsubj': 20, 'xcomp': 21, 'det:predet': 22, 'nmod:npmod': 23, 'acl': 24, 'amod': 25, 'expl': 26, 'csubjpass': 27, 'case': 28, 'ccomp': 29, 'dobj': 30, 'ROOT': 31, 'discourse': 32, 'nmod:tmod': 33, 'dep': 34, 'nummod': 35, 'mark': 36, 'advmod': 37, 'cc': 38, 'det': 39}
    dep_features = list()
    for w in dep_rels:
        if w in dep_dict:
            dep_feature = dep_dict[w]
        else:
            dep_feature = dep_dict["O"]
        dep_features.append(dep_feature)
    return dep_features


def get_sub_features(string, entity_subtype):
    #这个函数把entity_subtype从string映射成id
    #输入的entity_subtype是一个列表，这个列表装着句子每个词对应的entity_subtype的string
    #输出是一个列表，把输入的entity_subtype映射成id返回
#     entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_time': 2, '2_Group': 3, '2_Nation': 4, '2_Indeterminate': 5, '2_Population_Center': 6, '2_Government': 7, '2_Commercial': 8, '2_Non_Governmental': 9, '2_Media': 10, '2_Building_Grounds': 11, '2_Numeric': 12, '2_State_or_Province': 13, '2_Region_General': 14, '2_Sports': 15, '2_Crime': 16, '2_Land': 17, '2_Air': 18, '2_Water': 19, '2_Airport': 20, '2_Sentence': 21, '2_Educational': 22, '2_Celestial': 23, '2_Underspecified': 24, '2_Shooting': 25, '2_Special': 26, '2_Subarea_Facility': 27, '2_Path': 28, '2_GPE_Cluster': 29, '2_Exploding': 30, '2_Water_Body': 31, '2_Land_Region_Natural': 32, '2_Nuclear': 33, '2_Projectile': 34, '2_Region_International': 35, '2_Medical_Science': 36, '2_Continent': 37, '2_Job_Title': 38, '2_County_or_District': 39, '2_Religious': 40, '2_Contact_Info': 41, '2_Chemical': 42, '2_Subarea_Vehicle': 43, '2_Entertainment': 44, '2_Biological': 45, '2_Boundary': 46, '2_Plant': 47, '2_Address': 48, '2_Sharp': 49, '2_Blunt': 50}
    
    entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_Group': 2, '2_time': 3, '2_Nation': 4, '2_Building_Grounds': 5, '2_Government': 6, '2_Crime': 7, '2_Non_Governmental': 8, '2_Population_Center': 9, '2_Commercial': 10, '2_Region_General': 11, '2_Indeterminate': 12, '2_Media': 13, '2_Money': 14, '2_Air': 15, '2_Land': 16, '2_Path': 17, '2_Job_Title': 18, '2_Sentence': 19, '2_State_or_Province': 20, '2_Airport': 21, '2_Water': 22, '2_Exploding': 23, '2_Subarea_Facility': 24, '2_Underspecified': 25, '2_Sports': 26, '2_Projectile': 27, '2_Shooting': 28, '2_Celestial': 29, '2_Entertainment': 30, '2_Special': 31, '2_GPE_Cluster': 32, '2_Water_Body': 33, '2_Boundary': 34, '2_Land_Region_Natural': 35, '2_Educational': 36, '2_Region_International': 37, '2_Nuclear': 38, '2_Religious': 39, '2_Percent': 40, '2_Medical_Science': 41, '2_Continent': 42, '2_Subarea_Vehicle': 43, '2_Chemical': 44, '2_County_or_District': 45, '2_Biological': 46, '2_Sharp': 47, '2_Plant': 48, '2_Blunt': 49, '2_E_Mail': 50, '2_Address': 51, '2_Phone_Number': 52, '2_URL': 53}
    
    subtype_featrues = list()
    for w in entity_subtype:
        subtype_featrue = 0
        if w == "O":
            subtype_featrue = 0
        else:
            subtype_featrue = entity_subtype_dict[w[w.index("-")+1:].replace("-","_")]
            
            
        subtype_featrues.append(subtype_featrue)
    return subtype_featrues


def get_seg_features(string,tags):
    #定义了一个entity type到id的映射，tags原来装的是entity type的string，把它变成id然后返回，另外，这里不管B还是I的，无论是B-1_PER还是I-1_PER他们对应的id都是1
    tags_dict = {'O': 0, '1_PER': 1, '1_TIM': 2, '1_GPE': 3, '1_ORG': 4, '1_FAC': 5, '1_LOC': 6, '1_VEH': 7, '1_Numeric': 8, '1_WEA': 9, '1_Crime': 10, '1_Sentence': 11, '1_Job_Title': 12, '1_Contact_Info': 13}
    
    
    seg_feature = []
    entity_tag = 0
    for tag in tags:
        if "1_PER" in tag:
            entity_tag = 1
        elif "1_GPE" in tag:
            entity_tag = 2
        elif "1_Time" in tag:
            entity_tag = 3
        elif "1_ORG" in tag:
            entity_tag = 4
        elif "1_FAC" in tag:
            entity_tag = 5
        elif "1_VEH" in tag:
            entity_tag = 6
        elif "1_GPE" in tag:
            entity_tag = 7
        elif "1_Numeric" in tag:
            entity_tag = 8
        elif "1_Crime" in tag:
            entity_tag = 9
        elif "1_Sentence" in tag:
            entity_tag = 10
        elif "1_Contact_Info" in tag:
            entity_tag = 11
        elif "1_Job_Title" in tag:
            entity_tag = 12
        elif "1_WEA" in tag:
            entity_tag = 13
        else:
            entity_tag = 0
        seg_feature.append(entity_tag)
    return seg_feature

def get_pos_features(string ,pos):
    pos_dict = {'PAD' : 0, 'IN': 1, 'DT': 2, 'NNP': 3, 'JJ': 4, 'NNS': 5, ',': 6, 'PRP': 7, 'RB': 8, 'VBD': 9, '.': 10, 'VB': 11, 'CC': 12, 'VBN': 13, 'VBG': 14, 'VBP': 15, 'VBZ': 16, 'CD': 17, 'TO': 18, 'PRP$': 19, 'MD': 20, 'HYPH': 21, ':': 22, 'POS': 23, 'WP': 24, 'WDT': 25, 'RP': 26, 'UH': 27, 'WRB': 28, '``': 29, "''": 30, 'NNPS': 31, 'JJR': 32, '-RRB-': 33, '-LRB-': 34, 'EX': 35, 'JJS': 36, 'RBR': 37, 'NFP': 38, '$': 39, 'RBS': 40, 'PDT': 41, 'FW': 42, 'SYM': 43, 'WP$': 44, 'LS': 45, 'ADD': 46, 'AFX': 47, 'NN': 48}#加入PAD,VALUE是0
    pos_features = []
    for p in pos:
        if p in pos_dict:
            pos_features.append(pos_dict[p])
        else:
            print("wrong pos! "+string)
    return pos_features


class BatchManager(object):
#这个类接受的data是经过loader处理的data，一个句子相关的数据会用一个list来装起来，然后batch_size个句子的数据会被装在一个更外层的list里头
#经过这个类处理以后，self.batch_data是把数据按照句子长度划分成多个batch，所有batch的数据都储存在self.batch_data里面，对于每一个batch来说，它的形式是[strings, lens, chars, segs, subtypes, targets, adj, dep]
#strings是把data所有句子的字符串形成一个数组，lens是data所有句子的长度，chars是把strings中的字符串映射成id，segs是entity-type的id，subtypes是entity_subtype的id，targets是bio标签，adj是max_length*max_length的邻接矩阵，dep是max_length*max_length的依赖矩阵
#但是这个BatchManager最大的问题是它shuffle得不够彻底，它先按句子长度由大到小排列，然后按照batch_size去划分batch，当shuffle=True的时候，把划分出来的batch打乱顺序返回，感觉这样shuffle得并不是很彻底，因为每一个batch的内容在一开头就按照句子长度设定好了
#补充：这里按照句子长度由大到小排序是因为模型rnn那里，nn.utils.rnn.pack_padded_sequence方法默认输入数据的句子长度是由大到小排序的。
    def __init__(self, data, batch_size, num_steps):
        # data: string, doc_chars, chars, types, subtypes, tags

        self.batch_data = self.sort_and_pad(data, batch_size, num_steps)
        self.len_data = len(self.batch_data)
        self.length = int(num_steps)
    def sort_and_pad(self, data, batch_size, num_steps):
        self.num_batch = int(math.ceil(len(data) / batch_size))
        print("num_batch: ", self.num_batch)
        lens = [len(x[0]) for x in data] # 句子长度

        sorted_data = sorted(data, key=lambda x:len(x[0]), reverse=True)#把data按句子长度由大到小排列
        batch_data = list()
        for i in trange(self.num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size],num_steps))
        return batch_data

    @staticmethod
    def pad_data(data, length):
    #这个函数其实就是设定了一个max_length，如果句子长度不足max_length，那么跟他有关的所有数据都要用0填充到max_lenghth，如果句子长度比max_length大，那么只取前面长度为max_length的内容
    #返回的结果形式是[strings, lens, chars, segs, subtypes, targets, adj, dep]
    #这些数据都是加了padding的，会统一用数字0 padding到预设的max_length为止
    #strings是把data所有句子的字符串形成一个数组，lens是data所有句子的长度，chars是把strings中的字符串映射成id，segs是entity-type的id，subtypes是entity_subtype的id，targets是bio标签，adj是max_length*max_length的邻接矩阵，dep是max_length*max_length的依赖矩阵
        strings = []
        chars = []
        segs = []
        subtypes = []
        targets = []
        adj, dep = [], []
        lens = []
        poses = []
        max_length = length
        for line in data:
            string, char, seg, subtype, target, dep_rel_features, dep_word_idx, pos= line
            string_len = len(string)
            if string_len <= max_length:
                lens.append(string_len)
                padding = [0] * (max_length - len(string))
                strings.append(string + padding)#python的列表没有硬性要求一个list里面的元素类型都要相同，所以这里字符串数组string跟int数组padding是可以连起来的
                chars.append(char + padding)
                segs.append(seg + padding)
                targets.append(target + padding)
                subtypes.append(subtype + padding)
                poses.append(pos + padding)
            else:
                lens.append(max_length)
                strings.append(string[0:max_length])
                chars.append(char[0:max_length])
                targets.append(target[0:max_length])
                segs.append(seg[0:max_length])
                subtypes.append(subtype[0:max_length])
                poses.append(pos[0:max_length])

            # Dep:
            curr_adj = np.eye(max_length)#生成max_length*max_length的对角矩阵，对角线为1，其他为0
            curr_dep = np.random.randint(0, 1, (max_length, max_length), dtype=int)#生成一个max_length*max_length的矩阵，里面的数是0
            for j,dep_relation in enumerate(dep_rel_features):
                if j >= max_length:
                    break
                token1_id, token2_id = j, int(dep_word_idx[j])
                if token2_id == -1 or token2_id >= max_length:
                    token2_id = token1_id
                curr_adj[token1_id, token2_id], curr_adj[token2_id, token1_id] = 1, 1
                curr_dep[token1_id, token2_id], curr_dep[token2_id, token1_id] = int(dep_relation), int(dep_relation)
            adj.append(curr_adj)
            dep.append(curr_dep)
        return [strings, lens, chars, segs, subtypes, targets, adj, dep, poses]

    def iter_batch(self, shuffle = False):
        #这是一个生成器，通过yield来返回一个可迭代对象
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def input_from_line(line, char_to_id):
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


