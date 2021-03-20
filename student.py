'''
student.py
Asha Karki 
Date: 18/11/2020
UNSW COMP9444 Neural Networks and Deep Learning
'''

import re
import string
import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#import sklearn
import nltk
#from config import device

#nltk.download('stopwords')
################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################
#stopWords = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))

stopWords={'m', 'ma', 'nor', '$', 'after', 'what', 'have', 'hasn', 'before', "doesn't", 'about', 'did', '_', 'yourself', 'to', "needn't", "you'd", "weren't", 'are', 'under', "don't", 'he', 'shouldn', 'own', '>', 'him', 'or', 'some', 'against', 'd', 'how', 'on', 'too', 'mightn', "it's", "you're", 'itself', 'themselves', 'wasn', 'been', 'more', 'we', '-', 'other', 'both', '.', 'you', 'me', 'if', 'again', '/', 'during', 'should', 'o', 'himself', 'were', 're', 'those', 'myself', '"', 'than', 'y', 'which', 'll', '#', 'off', 'having', 'by', 'for', "mustn't", 'from', '`', 'between', 'am', 'won', 'into', '[', 'ours', 'your', 'out', 'yourselves', 'don', "mightn't", 'down', 'a', 'at', 'was', 'ourselves', '(', 'but', 'yours', 'these', "you've", '}', "won't", 'with', "didn't", 'is', 'i', 'who', 'few', "'", 'doing', '|', "aren't", "hasn't", 'because', 'where', 'that', 'shan', 'wouldn', ':', 'herself', '%', 'an', 'as', 'isn', ')', '*', 'hers', 'over', 'through', '{', 'not', 'while', 'can', "should've", 'haven', "wasn't", '?', "shan't", "isn't", 'the', 'needn', 'didn', 'in', "she's", '&', 'them', 'do', 'when', 'it', 'our', 'my', 'just', "shouldn't", '\\', 'above', 'hadn', '^', 'until', ']', 'this', 'further', 'then', 'below', 'same', ',', '!', 'doesn', 'ain', 'her', 'only', 'they', 'his', 'has', 's', 'aren', '~', 'no', "hadn't", '+', 'all', 'theirs', 'she', 'now', 'any', "haven't", "wouldn't", 'its', 'here', 'once', 've', 'mustn', 'each', 'had', 'couldn', 't', '=', 'most', 'being', 'weren', '@', 'of', 'there', 'will', ';', 'whom', 'and', "that'll", 'so', 'their', '<', 'does', 'why', 'be', 'up', "couldn't", "you'll", 'very', 'such'}


wordVectors = GloVe(name='6B', dim=300)


def tokenise(text):
    """
    Called before any processing of the text has occurred.
    """
    
    tokenize = []
    for word in nltk.casual_tokenize(text, preserve_case=False):
        # check stopwords and numbers ------ filter at this stage
        if word not in stopWords and not word.isnumeric():
            tokenize.append(word)
    
    #tokens = text.split()
    return tokenize



def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # join all the tokens with space - easy to process together
    input = " ".join(sample)
    #print(input)
    
    # replace non-ASCII characters in sample with spaces
    text = re.sub(r"[^\x00-\x7F]+", " ", input)
    #print(text)
    
    # removing numbers, special characters, then convert the tokens to lower case
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())
    
    #print(nopunct)
    
    # now converting back to the list of tokens -- words 
    result = nopunct.split(" ")
    
    #print(result)
    
    # remove spaces and empty tokens if present
    result = filter(lambda x: x != ' ', result)
    result = list(filter(lambda x: x != '', result))
        
    return result



def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    
    # Removing infrequent words from the batch
    # freqs – A collections.Counter object holding the frequencies of tokens 
    # in the data used to build the Vocab.
    count_vocab = vocab.freqs
    
    # itos – A list of token strings indexed by their numerical identifiers.
    itos_vocab = vocab.itos
    
    for words in batch:
        for i, word in enumerate(words):
            
            if count_vocab[itos_vocab[word]] < 3:
                words[i] = -1
 
    
    return batch



################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
        
    ratingOutput_p = torch.round(ratingOutput)
    _, categoryOutput_p = torch.max(categoryOutput, 1)
    
    
    return ratingOutput_p.long(), categoryOutput_p

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()

        # Model parameters
        self.hidden_dim = 128
        num_layers = 1
        drop_rate = 0.1
        vocab_size = 300
        
        # biLSTM layer
        self.lstm = tnn.LSTM(input_size= vocab_size, hidden_size=self.hidden_dim, bidirectional=True, num_layers=num_layers, batch_first=True)
        #self.lstm = tnn.GRU(vocab_size, hidden_dim, num_layers, batch_first=True, dropout=drop_rate)
        
        # Fully connecte layers 
        self.linear1 = tnn.Linear(in_features= 2 * self.hidden_dim, out_features=1)
        self.linear2 = tnn.Linear(in_features= 1 * self.hidden_dim, out_features=5)
        
        self.dropout = tnn.Dropout(drop_rate)
        
        # output layers
        self.softmax = tnn.Softmax(dim=1)
        self.sigmoid = tnn.Sigmoid()
        
    
    def forward(self, input, length):
        
        # required to avoid errors due to pad tokens when size length goes less than 0       
        for i, _ in enumerate(length):  
            if length[i] == 0:
                length[i] = 1
                
        # get the packed input ----
        packed_input = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
              
        # biLSTM layer --------
        output, (hidden, _) = self.lstm(packed_input)
        
        
        # for output1: Rating ---------
        output, _ = pad_packed_sequence(output, batch_first=True)
        outf = output[range(len(output)), length - 1, :self.hidden_dim]
        outr = output[:, 0, self.hidden_dim:]
        out_r = torch.cat((outf, outr), 1)
        output_after = self.dropout(out_r)
        
        
        # FC layer ---------
        rating_output = self.linear1(output_after)
        rating_output = torch.squeeze(rating_output, 1)
        rating_output = self.sigmoid(rating_output)
        
        
        # output 2: Business category ----------
        category_output = self.linear2(hidden[-1])
        category_output = self.softmax(category_output)
        
        
        return rating_output, category_output



class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss_fnc1 = tnn. BCELoss()
        self.loss_fnc2 = tnn. CrossEntropyLoss()
        
        
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
                
        ratingOutput = torch.flatten(ratingOutput)
        loss1 = self.loss_fnc1(ratingOutput.float(), ratingTarget.float())
        
        loss2 = self.loss_fnc2(categoryOutput, categoryTarget)
        
        loss = loss1 + loss2
        return loss
   


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.9
batchSize = 64
epochs = 10
#optimiser = toptim.SGD(net.parameters(), lr=0.01, momentum =0.9)
optimiser = toptim.Adam(net.parameters(), lr= 0.001)
