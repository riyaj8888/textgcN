


import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import re
import string as strg
from collections import defaultdict
import nltk,torch
from collections import Counter
from sklearn.feature_extraction import text
from math import log
import scipy.sparse as sp
from tqdm import tqdm
import scipy.sparse as sparse
import scipy.io as sio
import scipy.stats as stats
from torch_geometric.data import Data
import os
torch.manual_seed(42)
from sklearn import metrics
import matplotlib.pyplot as plt

## read a dataframe which has columns as "text","target str"
text_raw = pd.read_csv("data_IE_topic.csv")
label_dict = {label:i for i,label in enumerate(list(text_raw['target'].unique()))}
text_raw['label'] = text_raw['target'].map(label_dict)
text_raw_shuffle= text_raw.sample(frac=1)

## clean documents
def clean_doc(string):

    string = str(string)
    string = re.sub(r"^\"", "", string)
    string = re.sub(r"\"$", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\b(?<![0-9-])(\d+)(?![0-9-])\b"," ",string)
    string = re.sub("<\s?(\w[^A-Za-z0-9]?)+>", " ", string)
    string = re.sub(r'^https?:\/\/.*[\r\n]*', '', string, flags=re.MULTILINE)
    string = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", string)
    string = re.sub('\w*\d\w*', '', string)
    string = re.sub('[%s]' % re.escape(strg.punctuation), '', string)

    return string.strip().lower()

#building word frequency

def build_word_freq(text_list):

    word_freq = defaultdict(int)

    for doc_words in text_list:

        words = doc_words.split()
        for word in words:
            word_freq[word] += 1

    vocab = list(set(word_freq.keys()))
    
    vocab_size = len(vocab)

    # word to index dictionary
    word_id_map = {}

    for i in range(vocab_size):

        word_id_map[vocab[i]] = i

    return word_freq ,vocab,vocab_size,word_id_map

# word doc frequnecy

def Word_doc_Freq(text_list):

    word_doc_list = {}

    for i in range(len(text_list)):

        doc_words = text_list[i]
        words = doc_words.split()

        appeared = set()

        for word in words:

            if word in appeared:
                continue

            if word in word_doc_list:

                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list

            else:
                word_doc_list[word] = [i]
                
            appeared.add(word)


    word_doc_freq = {}

    for word, doc_list in word_doc_list.items():

        word_doc_freq[word] = len(doc_list)
    
    return word_doc_freq,appeared


# filter words based on word freq
def remove_words(sentences_raw,min_freq=5):
    
    clean_docs = []

    for doc_content in sentences_raw:

        temp = clean_doc(doc_content)
        words = temp.split()
        doc_words = []
        
        for word in words:

            if word not in stop_words and word_freq[word] >= min_freq:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)

    return clean_docs


# creating train ,val test masks


def create_masks(text_clean_shuffle,vocab,train_val_test_split_ratio = (0.6,0.2,0.2)):

    docs_vocabs_text = list(text_clean_shuffle['text'].values)+vocab

    doc_labels = text_clean_shuffle['label'].values.tolist()

    new_labels = doc_labels+[max(doc_labels)+1]*len(vocab)

    DF = pd.DataFrame({"text":docs_vocabs_text,"label":new_labels})

    unique_labels = list(DF['label'].unique())# can be removed above df and this line
    
    tr,v,ts = train_val_test_split_ratio

    train_ids = []
    
    val_ids = []

    test_ids = []

    #iterate over all lables in dataset
    for label in unique_labels:

        # collect idx position of all samples for each label
        index_ = list(DF[DF['label']==label].index)
        # size of samples for that label
        N = len(index_)
        
        if label != max(doc_labels)+1:

            ##get 60% of total samples for train
            st = int(N*tr)
            #get 20% of total samples for val
            sv = int(N*v)
            #get 20% of total samples for test
            stx = int(N*ts)

            # st = 45
            # sv = 25
            
            # collect samples upto 80% for training
            t_idx = index_[:st]
            # collect samples upto 20% for val
            v_idx = index_[st:(st+sv)]
            # collect samples upto 20% for test
            ts_idx = index_[st+sv:N+1]

            train_ids.extend(t_idx)

            val_ids.extend(v_idx)

            test_ids.extend(ts_idx)

        # else:

        #     # collect samples upto 80% for training
        #     # t_idx = index_[:500]
        #     # collect samples upto 20% for val
        #     # v_idx = index_[:750]
        #     # collect samples upto 20% for test
        #     ts_idx = index_[:N+1]

        #     # train_ids.extend(t_idx)
        #     # val_ids.extend(v_idx)
        #     test_ids.extend(ts_idx)


    return train_ids,val_ids,test_ids,new_labels

## building edges between word-word and doc-word
def build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    # constructing all windows
    windows = []
    for doc_words in doc_list:
        words = doc_words.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                
                # try :
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1

                # except KeyError:
                #     pass

    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:

            # try :
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

            
            # except KeyError:
            #     pass

    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue

            # try :

            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                    word_doc_freq[vocab[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)
            
            # except KeyError:
            #     pass

    number_nodes = num_docs + len(vocab)

    ### nodes from 0:num_docs belongs to doc nodes and num_docs:number_nodes belongs to word-word(vocab)
    ## here block matrix of size (vocab x vocab) belongs to word-word edge/graph.
    ## remaining (number_nodes-vocab x number_nodes-vocab) block corresponds to edges of documents graph.
    ## first nodes are for doc nodes and remaining are for word nodes
    
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)

    return adj

##glove word embed assignment

def assigne_glove_embed(text_clean_shuffle,vocab):

    total_nodes = text_clean_shuffle.shape[0]+len(vocab)

    docs_vocabs_text = list(text_clean_shuffle['text'].values)+vocab

    features = np.empty(shape=(total_nodes,word_embeddings_dim))

    for i in tqdm(range(0, len(docs_vocabs_text)), desc ="glove word embeddings assigning->"):

        sent = docs_vocabs_text[i]

        if len(sent)>1:

            features[i] = np.random.uniform(-0.01, 0.01,300)

        else:

            if sent in list(word_embeddings.keys()):

                features[i] = word_embeddings[sent]#np.random.uniform(-0.01, 0.01,300)

            features[i] = np.random.uniform(-0.01, 0.01,300)

    return features

### stop word removal
stop = text.ENGLISH_STOP_WORDS

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

stop_words_full = list(stop_words) + list(stop)



# read raw sentences
sentences_raw = text_raw_shuffle['text'].values
#create word freq ,vocab and word2id map
word_freq ,vocab,vocab_size,word_id_map = build_word_freq(sentences_raw)
#remove words with min_freq as occurrence threshold
sentences_clean = remove_words(sentences_raw,min_freq=5)
#create new df for cleaned sentences
text_clean_shuffle = pd.DataFrame({"text":sentences_clean,"label":text_raw_shuffle['label'].values.tolist()})
text_clean_shuffle['text'] = text_clean_shuffle['text'].apply(lambda x: ' '.join(item for item in x.split() if item not in stop_words_full))
#build word freq,vocab for cleaned sentences df
word_freq,vocab,vocab_size,word_id_map = build_word_freq(text_clean_shuffle['text'].values)
#create word doc freq for tf-idf calculation
word_doc_freq,appeared = Word_doc_Freq(text_clean_shuffle['text'].values)
# creating adj matrix for graph construction
adj = build_edges(text_clean_shuffle['text'].values, word_id_map, vocab, word_doc_freq, window_size=20)
# write adjacency matrix so we can use it directly next time
sio.mmwrite("sparse_adj_matrix_IEngg_new_chunks.mtx",adj)
# creating train,val and test mask
train_idx,val_idx,test_idx,new_labels = create_masks(text_clean_shuffle,vocab)
#total number of nodes
N = text_clean_shuffle.shape[0]+vocab_size
#set the mask
train_mask = np.zeros((N,),dtype=bool)
train_mask[train_idx] = True

val_mask = np.zeros((N,),dtype=bool)
val_mask[val_idx] = True

test_mask = np.zeros((N,),dtype=bool)
test_mask[test_idx] = True

train_mask = torch.tensor(train_mask)
val_mask = torch.tensor(val_mask)
test_mask = torch.tensor(test_mask)
#snetences and vocab nodes labels
y = torch.tensor(new_labels).to(torch.long)
#check distribution of train.val and test labels
print(Counter(np.array(new_labels)[train_mask]))
print(Counter(np.array(new_labels)[val_mask]))
print(Counter(np.array(new_labels)[test_mask]))

print("total samples:",y.shape)
print("train samples:",train_mask.sum())
print("val samples:",val_mask.sum())
print("test samples:",test_mask.sum())


# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r',encoding="utf8") as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float,data[1:]))

features = assigne_glove_embed(text_clean_shuffle,vocab)

# create torch geo dataset 

A = adj.tocoo()#adj_.tocoo()
row = torch.from_numpy(A.row).to(torch.long)
col = torch.from_numpy(A.col).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
edge_weight = torch.from_numpy(A.data).to(torch.float)


        
X = torch.tensor(features,dtype=torch.float)  # featureless
Y = y
data = Data(x = X, edge_index=edge_index, edge_attr=edge_weight, y=Y,train_mask = train_mask,val_mask = val_mask,test_mask = test_mask )


# define gcn model

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = torch.nn.Linear(self.in_channels, self.out_channels)
        self.lin_r = torch.nn.Linear(self.in_channels, self.out_channels)


    def forward(self, x, edge_index):

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        
        out = self.propagate(edge_index, x=x, norm=norm)
        #skip connection
        out = self.lin_l(x) + self.lin_r(out)

        return out

    def message(self, x_j, norm):
        
        # x_j has shape [E, out_channels]

        return norm.view(-1, 1) * x_j


import torch.nn.functional as F
class GCN(torch.nn.Module):
    def __init__(self, in_channels, NUM_CLASS):
        super(GCN, self).__init__()

        self.lin1 = torch.nn.Linear(in_channels, 16)
        self.lin2 = torch.nn.Linear(16, 32)

        self.conv1 = GCNConv(32, 64) 
        self.conv2 = GCNConv(64, 128) 

        self.lin3= torch.nn.Linear(128, 192)
        self.lin4 = torch.nn.Linear(192, NUM_CLASS)

    def forward(self, x, edge_index):


        x = self.lin1(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25,training=self.training)


        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.25,training=self.training)

        # x = self.conv1(x, edge_index).relu()
        # x = F.dropout(x, p=0.5,training=self.training)

        # x = self.conv2(x, edge_index).relu()
        # x = F.dropout(x, p=0.5,training=self.training)
        
        x = self.lin3(x)
        x = self.lin4(x)

        return x

input_features = X.shape[1]
num_class = y.max()+1
model_gcn = GCN(input_features,num_class)
print(model_gcn)


# training loop 
# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #'cpu'#

model = model_gcn.to(device)
data = data.to(device)
# from sklearn import metrics
# Initialize Optimizer
learning_rate = 0.005
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate,weight_decay=decay)
                             #)
# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x,data.edge_index)  

      pred = out.argmax(dim=1)  
      # print(pred,data.y[data.train_mask])
      # Check against ground-truth labels.
      train_correct = pred[data.train_mask] == data.y[data.train_mask]

      # print(metrics.classification_report(y_true,y_pred))

      train_acc = int(train_correct.sum()) / int(data.train_mask.sum()) 

      val_correct = pred[data.val_mask] == data.y[data.val_mask]
      val_acc = int(val_correct.sum()) / int(data.val_mask.sum()) 

      y_true = data.y[data.val_mask].detach().cpu()
      y_pred = pred[data.val_mask].detach().cpu()

      # fs = f1_score(y_true,y_pred,average=None)
      # prec = precision_score(y_true,y_pred,average=None)
      # rec = recall_score(y_true,y_pred,average=None)

      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      # print(loss)
      loss.backward() 
      optimizer.step()
      # print(pred[data.train_mask].shape)
      return loss,train_acc,val_acc,y_true,y_pred

def test():
      model.eval()
      out = model(data.x,data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())


      y_true = data.y[data.test_mask].detach().cpu()
      y_pred = pred[data.test_mask].detach().cpu()

      # fs = f1_score(y_true,y_pred,average=None)
      # prec = precision_score(y_true,y_pred,average=None)
      # rec = recall_score(y_true,y_pred,average=None)

      return test_acc,y_true,y_pred

labels_ = list(label_dict.keys())
losses = []
for epoch in range(0,5000):

      loss,train_acc,val_acc,y_true,y_pred = train()
      
      losses.append(loss.item())

      if epoch % 50 == 0:
            
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},train_acc:{train_acc},val_acc:{val_acc}')
            print("\n")
            print("validation Precision, Recall and F1-Score...")
            print(metrics.classification_report(y_true,y_pred, digits=4))
            print("Macro average validation Precision, Recall and F1-Score...")
            print(metrics.precision_recall_fscore_support(y_true,y_pred, average='macro'))
            print("Micro average validation Precision, Recall and F1-Score...")
            print(metrics.precision_recall_fscore_support(y_true,y_pred, average='micro'))
            



import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training loss curve")
plt.savefig("training_curve_gcn.png")

# testing gcn on test dataset
test_acc,y_true,y_pred = test()


print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(y_true,y_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(y_true,y_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(y_true,y_pred, average='micro'))


torch.save("model_gcn.pt",model)