# Rating-and-category-prediction-
This work is a part of COMP9444, Neural Networks and Deep Learning @UNSW 2020


This project implements a python program (using pytorch library) that learns to read business reviews in text format and predict a rating (positive or negative) associated with each review and a business category (0=Restaurants, 1=Shopping, 2=Home Services, 3=Health & Medical, 4=Automotive).

Highlights:
-- Processed basic text data in a JSON format, used tokenization, preprocessing, and postprocessing
-- Used bidirectional LSTM model for learning and inference



# Text processing

## Tokenize:
This is the first step to convert text-based data to smaller chunks called tokens. Here the tokens are words. We use nltk.casual_tokenize() function to tokenize the text. Then we collect the tokens that are not stopwords and numbers -- we applied this filtering at this stage. We tried simple text.split() function as well, but we got better tokens with nltk's function.  
    
## Preprocessing:
We remove the special characters, space, and numbers if present in all tokens in this process. We use regular python expressions (re) for this purpose. Then we convert all tokens to lowercase. Our steps are the following:
1) Join all the tokens with space - easy to process together
2) Replace non-ASCII characters in the sample with spaces
3) Remove numbers, special characters, then convert the tokens to lower case
4) Convert back to the list of tokens
5) Remove spaces and empty tokens if present

## Postprocessing:
In this process, we remove infrequent words that are less than three from each batch. We tried different values, but kept three as we observed better performance. Torchtext vocab .freqs and .itos are used for this task. "freqs" is an object holding the frequencies of tokens in the data used to build the vocab, and "itos" is a list of token strings indexed by their numerical identifiers.
 
Note: Tokenize, Preprocessing a Postprocessing completes our text processing part.



# Machine learning Model

## Network:
In this work, we use a bidirectional LSTM model with  fully connected linear network layers: biLSTM --> fully connected layer --> output layer. 

We use Long short-term memory (LSTM) because it can process entire sequences of text and well-known classifier for the classification problem with the text data. Moreover, it can learn order dependence in a sequence it is designed specially to deal with the vanishing gradient problem in Recursive neural networks. LSTM cell vectors have both forgetting parts and adding new information parts. We tried with various other models based on LSTM and Gated recurrent units (GRU) but we got the best result using biLSTM with the hidden dimension of 128. As we have two outputs, one for the rating (two-class) and another for the business category (five classes), we
used different linear network layers for these two outputs: For the binary output, we used the sigmoid function as the output layer, and for the business category output, we used the softmax function as the output layer. Softmax function is used for the multiclass classification function that provides the probability of  each class.

Overall, our model is the following where biLSTM is common for  both outputs 1) and 2) below:
1) For rating: biLSTM --> fully connected  --> sigmoid (output 1: binary)
2) For categories: biLSTM --> fully connected --> softmax (output 2: 5 classes) 

In our experiment, we tried with one and two layers of biLSTM, but we got the best result with 
one layer of biLSTM.

## Hyperparameters: 
We find the following provides the best performance based on our observations:
Learning rate = 0.001 (accuracy drops by 3% if we use 0.0001, tried 0.1, 0.00001)
Batch size = 64 (did not observe much changes in the performance by varying batch size, tried 128, 32)
epochs = 10 (observed that loss does not drop significantly after epochs 10 -- achieved minima by 10 -- loss stays around 1)
dropout rate = 0.1 (to regularize over-fitting, tried 0.2)
optimizer = Adam (tried SGD, but Adam provides better accuracy)
train/val split  = 90:10 (tried 90:20)


## Loss functions:
For binary output, we use the Binary cross-entropy loss function as our output is binary. 
For business category output, we use the cross-entropy loss as our output is 
multiple classes. The overall loss function is the sum of these two functions.


# Helper function

## convertNetOutput():
In this function, we change the representation of our two outputs. As the ratingoutput
is one dimensional, we take its round as our output. Whereas, for the categoryOutput,
we have five outputs, so we choose the class that has the maximum output.  

Torchtext GloVe 6B word vectors are used for pretrained Word Embeddings. We used vocab
size 300, 200, 100, and 50, but got the best result with 300.


# Results
We observed the validation accuracy of around 85%, and
Rating incorrect, business category incorrect: 0.94%
Rating correct, business category incorrect: 12.76%
Rating incorrect, business category correct: 5.34%
Rating correct, business category correct: 80.96%



# Files:
1.	train.json: Dataset
2.	student.py: My python implementation
3.	helper files: hw2main.py and config.py




