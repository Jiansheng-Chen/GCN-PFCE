import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

# entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_Time': 2, '2_Group': 3, '2_Nation': 4,
#                        '2_Indeterminate': 5, '2_Population_Center': 6, '2_Government': 7,
#                        '2_Commercial': 8, '2_Non_Governmental': 9, '2_Media': 10, '2_Building_Grounds': 11,
#                        '2_Numeric': 12, '2_State_or_Province': 13, '2_Region_General': 14, '2_Sports': 15,
#                        '2_Crime': 16, '2_Land': 17, '2_Air': 18, '2_Water': 19, '2_Airport': 20,
#                        '2_Sentence': 21, '2_Educational': 22, '2_Celestial': 23, '2_Underspecified': 24,
#                        '2_Shooting': 25, '2_Special': 26, '2_Subarea_Facility': 27, '2_Path': 28,
#                        '2_GPE_Cluster': 29, '2_Exploding': 30, '2_Water_Body': 31, '2_Land_Region_Natural': 32,
#                        '2_Nuclear': 33, '2_Projectile': 34, '2_Region_International': 35, '2_Medical_Science': 36,
#                        '2_Continent': 37, '2_Job_Title': 38, '2_County_or_District': 39, '2_Religious': 40,
#                        '2_Contact_Info': 41, '2_Chemical': 42, '2_Subarea_Vehicle': 43, '2_Entertainment': 44,
#                        '2_Biological': 45, '2_Boundary': 46, '2_Plant': 47, '2_Address': 48, '2_Sharp': 49,
#                        '2_Blunt': 50
#                        }
entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_Group': 2, '2_time': 3, '2_Nation': 4, '2_Building_Grounds': 5, '2_Government': 6, '2_Crime': 7, '2_Non_Governmental': 8, '2_Population_Center': 9, '2_Commercial': 10, '2_Region_General': 11, '2_Indeterminate': 12, '2_Media': 13, '2_Money': 14, '2_Air': 15, '2_Land': 16, '2_Path': 17, '2_Job_Title': 18, '2_Sentence': 19, '2_State_or_Province': 20, '2_Airport': 21, '2_Water': 22, '2_Exploding': 23, '2_Subarea_Facility': 24, '2_Underspecified': 25, '2_Sports': 26, '2_Projectile': 27, '2_Shooting': 28, '2_Celestial': 29, '2_Entertainment': 30, '2_Special': 31, '2_GPE_Cluster': 32, '2_Water_Body': 33, '2_Boundary': 34, '2_Land_Region_Natural': 35, '2_Educational': 36, '2_Region_International': 37, '2_Nuclear': 38, '2_Religious': 39, '2_Percent': 40, '2_Medical_Science': 41, '2_Continent': 42, '2_Subarea_Vehicle': 43, '2_Chemical': 44, '2_County_or_District': 45, '2_Biological': 46, '2_Sharp': 47, '2_Plant': 48, '2_Blunt': 49, '2_E_Mail': 50, '2_Address': 51, '2_Phone_Number': 52, '2_URL': 53}

dep_dict = {'O': 0, 'punct': 1, 'iobj': 2, 'parataxis': 3, 'auxpass': 4, 'aux': 5,
            'conj': 6, 'advcl': 7, 'acl:relcl': 8, 'nsubjpass': 9, 'csubj': 10, 'compound': 11,
            'compound:prt': 12, 'mwe': 13, 'cop': 14, 'neg': 15, 'nmod:poss': 16, 'appos': 17,
            'cc:preconj': 18, 'nmod': 19, 'nsubj': 20, 'xcomp': 21, 'det:predet': 22,
            'nmod:npmod': 23, 'acl': 24, 'amod': 25, 'expl': 26, 'csubjpass': 27, 'case': 28,
            'ccomp': 29, 'dobj': 30, 'ROOT': 31, 'discourse': 32, 'nmod:tmod': 33, 'dep': 34,
            'nummod': 35, 'mark': 36, 'advmod': 37, 'cc': 38, 'det': 39
            }
pos_dict = {'PAD' : 0, 'IN': 1, 'DT': 2, 'NNP': 3, 'JJ': 4, 'NNS': 5, ',': 6, 'PRP': 7, 'RB': 8, 'VBD': 9, '.': 10, 'VB': 11, 'CC': 12, 'VBN': 13, 'VBG': 14, 'VBP': 15, 'VBZ': 16, 'CD': 17, 'TO': 18, 'PRP$': 19, 'MD': 20, 'HYPH': 21, ':': 22, 'POS': 23, 'WP': 24, 'WDT': 25, 'RP': 26, 'UH': 27, 'WRB': 28, '``': 29, "''": 30, 'NNPS': 31, 'JJR': 32, '-RRB-': 33, '-LRB-': 34, 'EX': 35, 'JJS': 36, 'RBR': 37, 'NFP': 38, '$': 39, 'RBS': 40, 'PDT': 41, 'FW': 42, 'SYM': 43, 'WP$': 44, 'LS': 45, 'ADD': 46, 'AFX': 47, 'NN': 48}

class EDModel(nn.Module):

    def __init__(self, args, id_to_tag, device, pre_word_embed):
        super(EDModel, self).__init__()

        self.device = device
        self.gcn_model = EEGCN(device, pre_word_embed, args)
        self.gcn_dim = args.gcn_dim
        self.classifier = nn.Linear(self.gcn_dim, len(id_to_tag))

    def forward(self, word_sequence, x_len, entity_type_sequence, adj, dep, pos):
        outputs, weight_adj = self.gcn_model(word_sequence, x_len, entity_type_sequence, adj, dep, pos)
        logits = self.classifier(outputs)
        return logits, weight_adj


class EEGCN(nn.Module):
    def __init__(self, device, pre_word_embeds, args):
        super().__init__()

        self.device = device
        self.in_dim = args.word_embed_dim + args.bio_embed_dim
        self.maxLen = args.num_steps

        self.rnn_hidden = args.rnn_hidden
        self.rnn_dropout = args.rnn_dropout
        self.rnn_layers = args.rnn_layers

        self.gcn_dropout = args.gcn_dropout
        self.num_layers = args.num_layers
        self.gcn_dim = args.gcn_dim

        # Word Embedding Layer
        self.word_embed_dim = args.word_embed_dim
        self.wembeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pre_word_embeds), freeze=False)

        # Entity Label Embedding Layer
        self.bio_size = len(entity_subtype_dict)
        self.bio_embed_dim = args.bio_embed_dim
        self.pos_size = len(pos_dict)
        self.pos_embed_dim = args.pos_embed_dim
        if self.bio_embed_dim:
            self.bio_embeddings = nn.Embedding(num_embeddings=self.bio_size,
                                               embedding_dim=self.bio_embed_dim)

        self.dep_size = len(dep_dict)
        self.dep_embed_dim = args.dep_embed_dim
        self.edge_embeddings = nn.Embedding(num_embeddings=self.dep_size,
                                            embedding_dim=self.dep_embed_dim,
                                            padding_idx=0)
        self.pos_embeddings= nn.Embedding(num_embeddings=self.pos_size, embedding_dim=self.pos_embed_dim, padding_idx=0)
        
        
        self.rnn = nn.LSTM(self.in_dim, self.rnn_hidden, self.rnn_layers, batch_first=True, \
                           dropout=self.rnn_dropout, bidirectional=True)
        self.rnn_drop = nn.Dropout(self.rnn_dropout)  # use on last layer output

        self.input_W_G = nn.Linear(self.rnn_hidden * 2, self.gcn_dim)
        self.pooling = args.pooling
        self.gcn_layers = nn.ModuleList()
        self.gcn_drop = nn.Dropout(self.gcn_dropout)
        self.pos_gates = nn.ModuleList()
        for i in range(self.num_layers+1):
            self.pos_gates.append(nn.Linear(self.pos_embed_dim, self.gcn_dim))
        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(self.device, self.gcn_dim, self.dep_embed_dim, args.pooling))
        self.aggregate_W = nn.Linear(self.gcn_dim + self.num_layers * self.gcn_dim, self.gcn_dim)#这里的input_dim是self.gcn_dim + self.num_layers * self.gcn_dim是因为聚合的时候，会把未经gcn变换的向量，和经过gcn变换的相连做聚合操作，所以要在self.num_layers * self.gcn_dim的基础上加多一个self.gcn_dim

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        # seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.rnn_hidden, self.rnn_layers)
        h0, c0 = h0.to(self.device), c0.to(self.device)
        #nn.utils.rnn.pack_padded_sequence其实是针对padding后的数据输入rnn的特殊处理，可见https://blog.csdn.net/m0_46483236/article/details/124136437
        #通过这个处理，padding 0在rnn之后得到的还是0；如果没有这个处理，因为rnn输出的h会考虑之前的状态，输入padding 0得到的不一定是0，这样就会使得rnn对数据的表示不准确，干扰实验效果
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.to('cpu'), batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, word_sequence, x_len, entity_type_sequence, adj, edge, pos):

        BATCH_SIZE = word_sequence.shape[0]
        BATCH_MAX_LEN = x_len[0]#因为BatchManager得到的每一个batch都是按句子长度由大到小排序的，x_len其实装的是这个batch所有句子的长度（这个长度不会超过预设的max_length），那么x_len[0]就是这个batch里面最长句子的长度
        #print(x_len)
        #contiguous()相当于深拷贝，以下三个语句，是把word_sequence、adj、edge中BATCH_MAX_LEN前的部分取出来,让第一个数据没有padding，其他数据后面可能有padding，这样一来，如果BATCH_MAX_LEN小于处理数据时预设的max_length的时候，就能进一步减少padding的数量
        word_sequence = word_sequence[:, :BATCH_MAX_LEN].contiguous()
        adj = adj[:, :BATCH_MAX_LEN, :BATCH_MAX_LEN].contiguous()
        edge = edge[:, :BATCH_MAX_LEN, :BATCH_MAX_LEN].contiguous()
        pos = pos[:, :BATCH_MAX_LEN].contiguous()
        weight_adj = self.edge_embeddings(edge)  # [batch, seq, seq, dim_e]
        pos_emb = self.pos_embeddings(pos)
        word_emb = self.wembeddings(word_sequence)
        x_emb = word_emb
        if self.bio_embed_dim:
            entity_type_sequence = entity_type_sequence[:, :BATCH_MAX_LEN].contiguous()
            entity_label_emb = self.bio_embeddings(entity_type_sequence)
            x_emb = torch.cat([x_emb, entity_label_emb], dim=2)

        rnn_outputs = self.rnn_drop(self.encode_with_rnn(x_emb, x_len, BATCH_SIZE))
        gcn_inputs = self.input_W_G(rnn_outputs)
        pos_gate = self.pos_gates[0](pos_emb)
        gcn_outputs = gcn_inputs
        gcn_list = pos_gate*gcn_inputs
        #gcn_list = gcn_outputs
        layer_list = [gcn_list]

        src_mask = (word_sequence != 0)#这里是对word_sequence每个元素进行是否等于0的判断，并把判断的结果Ture或False组成跟word_sequence一样形状的东西返回,在这里用的embedding 0是pad
        src_mask = src_mask[:, :BATCH_MAX_LEN].unsqueeze(-2).contiguous()
        #src_mask的形状batch_size*1*BATCH_MAX_LEN
        for _layer in range(self.num_layers):
            gcn_outputs, weight_adj = self.gcn_layers[_layer](weight_adj,gcn_outputs)  # [batch, seq, dim]
            pos_gate = self.pos_gates[_layer+1](pos_emb)
            gcn_list = self.gcn_drop(pos_gate*gcn_outputs)
            
            gcn_outputs = self.gcn_drop(gcn_outputs)
            weight_adj = self.gcn_drop(weight_adj)
            layer_list.append(gcn_list)
        
        outputs = torch.cat(layer_list, dim=-1)
        aggregate_out = self.aggregate_W(outputs)
        return aggregate_out, weight_adj


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()

        self.gcn_dim = gcn_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling

        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = Edgeupdate(gcn_dim, self.dep_embed_dim, dropout_ratio=0.5)
        self.ELU = nn.ELU()
    def forward(self, weight_adj, gcn_inputs):
        """
        :param weight_adj: [batch, seq, seq, dim_e]
        :param gcn_inputs: [batch, seq, dim]
        :return:
        """
        batch, seq, dim = gcn_inputs.shape
        weight_adj = weight_adj.permute(0, 3, 1, 2)  # [batch, dim_e, seq, seq]  dim_e就是self.dep_embed_dim，依赖的编码长度

        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.dep_embed_dim, seq, dim)#[batch, seq, dim]-->[batch, 1, seq, dim]-->[batch, self.dep_embed_dim, seq, dim]其实是把gcn_inputs复制了self.dep_embed_dim多份，然后把结果组织成[batch, self.dep_embed_dim, seq, dim]的形状（dim_e就是self.dep_embed_dim），方便后面的计算
        Ax = torch.matmul(weight_adj, gcn_inputs)  # [batch, dim_e, seq, dim]这是用边更新点的操作
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim(gcn_dim)]
        gcn_outputs = self.W(Ax)#[batch, seq, dim(gcn_dim)]
        #weights_gcn_outputs = F.relu(gcn_outputs)
        weights_gcn_outputs = self.ELU(gcn_outputs)
        node_outputs = weights_gcn_outputs
        # Edge update weight_adj[batch, dim_e, seq, seq]
        weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()  # [batch, seq, seq, dim_e]
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()#注意，node_outputs1和node_outputs2都是(batch_size,seq,seq,dim)，但是不同的是，node_outputs1第1维是复制扩展而来的，node_outputs2第2维是扩展而来的，为什么要这样做？这是因为后续要把两个点和边连接起来，所以要这样做
        #可以做以下实验进行验证，第一个点[1，2，3]，第二个点[4，5，6]，节点1和节点1的边是[111,112]，节点1和节点2的边是[121,122]，前面两个数字代表谁连接到谁，最后一个数字是边的维度（如111，就是节点1连接到节点1，维度1的值），以此类推。模仿Edge update的过程计算，最后结果r表示边和点正确连接起来了
#         import torch
#         t=torch.tensor([[1,2,3],[4,5,6]])
#         print('-----t-----')
#         print(t.shape)
#         print(t)
#         t1=t.unsqueeze(1).expand(2,2,3)
#         print('-----t1-----')
#         print(t1.shape)
#         print(t1)
#         t2=t1.permute(1, 0, 2).contiguous()
#         print('-----t2-----')
#         print(t2.shape)
#         print(t2)
#         t3=torch.cat([t1,t2],dim=-1)    
#         print('-----t3-----')
#         print(t3.shape)
#         print(t3)
#         dep=torch.tensor([[[111,112],[121,122]],[[211,212],[221,222]]])
#         print('-----dep-----')
#         print(dep.shape)
#         print(dep)
#         r=torch.cat([t3,dep],dim=-1)
#         print('-----r-----')
#         print(r.shape)
#         print(r)
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)
        return node_outputs, edge_outputs


class Edgeupdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.5):
        super(Edgeupdate, self).__init__()
        self.hidden_dim = hidden_dim#gcn_dim
        self.dim_e = dim_e#dep_embed_dim
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)

    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, seq, seq, dim_e]
        :param node: [batch, seq, seq, dim]
        :return:
        """
       #这里的node形状是[batch, seq, seq, dim]，可能是因为边是[batch, seq, seq, dim_e]这个形状，用点更新边的时候，需要把节点和边的信息拼接起来，然后再用权重进行变换，node设置成这个形状应该就是为了方便信息的拼接
       #但是node的形状怎么变成这样呢？还是说他一开始输入到模型就是这样呢？ 
        node = torch.cat([node1, node2], dim=-1) # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0, c0


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


