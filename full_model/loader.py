import os
import re
import codecs

from data_utils import iob2, iob_iobes, create_dico, create_mapping, get_seg_features, get_sub_features, get_dep_features, get_pos_features

def load_sentences(path):
    #把经过预处理成example_data样式的数据进行处理，load进来，返回一个列表sentences，它里面装着很多个sentence，每个sentence又是一个列表，list里面的元素是example_data里面每一行split之后的list
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line =line.rstrip()#把字符串两边的\t、\n、\r这些空白字符删除，返回得到的结果
        if not line:#因为句子之间会用一个空行来隔开，当遍历完一个句子的所有词以后，会把这个句子装到sentences里面，然后把sentence清空，再装下一个句子的词
            if len(sentence) > 0:
                sentences.append(sentence) 
                sentence = []
        else:#以下是处理一个一般行的情况
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word= line.split()
            
            sentence.append(word)
    if len(sentence) > 0:#这里是因为最后一个句子结束的时候不会有空行，所以不能用循环里面的if not line把sentence加到sentences里面，所以循环结束要再做判断sentence的长度是否大于0，如果是，应该把sentence加到sentences里面
        sentences.append(sentence)
    return sentences
#sentences里面装着很多个sentence，每个sentence又是一个list，list里面的元素是example_data里面每一行split之后的list


def update_tag_scheme(sentences, tag_scheme):
    #这个函数判断sentences里面储存的event-type label是否满足tag_scheme的要求，不是bio就抛出异常，event-type label是bio，tag_scheme也是bio就直接赋值返回，event-type label是bio，tag_scheme是bioes，就把标签的格式改成bioes再返回
    for i,s in enumerate(sentences):
        # tags = [w[-1] for w in s]
        tags = [w[4] for w in s]#根据README.md里面描述的数据格式，w[4]是event-type label
        if not iob2(tags):#iob2()判断tags是否属于bio标签，是返回True，否返回False这里是不属于bio标签的情况，直接抛出异常
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format!'
                            + 'please check sentence %i:\n%s' %(i,s_str))
        if tag_scheme == 'iob':#tag_scheme==iob的情况，直接赋值
            for word, new_tag in zip(s, tags):
                word[4] = new_tag
        elif tag_scheme == 'iobes':#tag_scheme==iobes的情况，要把bio转换成bioes再赋值
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[4] = new_tag
        else:
            raise Exception('Unknow tagging scheme!')

def char_mapping(sentences, lower):
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char

def augment_with_pretrained(dictionary,ext_emb_path,chars):
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
    ])
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id,id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def tag_mapping(sentences,index):#用index这一列的所有数据建立一个map
    tags = [[char[index] for char in s ] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def prepare_dataset(sentences, char_to_id, tag_to_id, char_to_id2 ,train = True):
    #整理数据格式，返回一个list，这个list每一个元素都是一个装着一个句子内容list
    #每一个句子的list的格式是[string, chars, types, subtypes, tags, dep_rel_features, dep_word_idx]
    #string是这个句子原来的样子，chars是把特殊字符都变成<UNK>，并且把词都变成id的序列，除了string以外都是id的形式
    none_index = tag_to_id['O']
    data = []

    for s in sentences: 
        string, entity_types, entity_subtype, tags, dep_rels, dep_word_idx, pos= list(), list(), list(), list(), list(), list(), list()
        for w in s:#这个循环是用来转换数据排列的格式，token->sentence entity_types等都变成一个list的形式，方便后续用来做序列标注
            if w[0] != "...":
                string.append(w[0]) # token --> sentence
                entity_types.append(w[2])
                entity_subtype.append(w[3])
                tags.append(w[4])
                dep_rels.append(w[5])
                pos.append(w[7])
            if w[6]=="O":#这是因为有一些词可能没有dep，使得他的依赖标签和gov都是英文字母O，这样会出错，所以遇到这种情况，把gov变成-1，到时候忽略掉
                dep_word_idx.append(-1)
            else:
                dep_word_idx.append(w[6])
        if len(string)> 4: 
            chars = [char_to_id[w.lower() if w.lower() in char_to_id else '[UNK]']
                     for w in string] #把特殊字符都变成[UNK]对应的id
            chars_ori = [char_to_id2[w if w in char_to_id2 else '<UNK>'] for w in string]
            types = get_seg_features(string, entity_types)  # convert to id
            subtypes = get_sub_features(string, entity_subtype) # convert to id
            dep_rel_features = get_dep_features(string, dep_rels)# convert to id
            pos = get_pos_features(string, pos)#convert to id
            if train:#如果是在为训练数据load的话返回正确的tags，如果是为测试数据load的话，返回的tags全部为0向量。这意味着，train和validate的时候，应该把train设置为True，而test的时候，应该把train设置为False
                tags = [tag_to_id[w.replace(":","_")] for w in tags]
            else:
                tags = [none_index for _ in chars]
            data.append([string, chars, types, subtypes, tags, dep_rel_features, dep_word_idx, pos, chars_ori])
    return data



