import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import itertools
import json

MAX_FEATURES = 30000
MAX_SEQUENCE_LENGTH = 300

class SequenceVectorize:
    def __init__(self, train_texts, calculate_vocab):
        '''
        calculate_vocab: Boolean, if True we calculate the vocabulary from the dataset.
        '''
            
        #Get vocabulary.
        if calculate_vocab==False:
            with open("PretrainedEmbedding/vocab.txt") as f:
                self.word_index = {}
                i = 1
                for line in f:
                    (key, _) = line.split()
                    self.word_index[key] = i
                    i+=1
                    if i == MAX_FEATURES + 1:
                        break
        elif calculate_vocab==True:
            tokenizer = text.Tokenizer()
            tokenizer.fit_on_texts(train_texts)
            self.word_index = dict(itertools.islice(tokenizer.word_index.items(), MAX_FEATURES))

    def vectorize(self, texts):
        vect_texts = []
        for t in texts:
            vect_text = []
            for w in t:
                try:
                    idx = self.word_index[w]
                except KeyError:
                    idx = 0
                vect_text.append(idx)
            vect_texts.append(vect_text)
        vect_texts = np.array(vect_texts, dtype=object)

        #Padding 
        vect_texts = sequence.pad_sequences(vect_texts, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
        return vect_texts

def onehot_encoding(labels):
    ''' 
    0: center
    1: left
    2: right
    '''
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(labels), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded, label_encoder
    
def onehot_decoding(onehot_encoded, label_encoder):
    onehot_decoded = label_encoder.inverse_transform([np.argmax(onehot_encoded)])
    return onehot_decoded

def train_val_test_split(vect_texts, labels, reproduceable):
    '''
    reproduceable: Boolean, if True we always get exactly the same shuffling
    '''
    if reproduceable == True:
        random_state = 20
    else:
        random_state = None
    
    #onehot_encoded, _ = onehot_encoding(labels)
    train_texts, test_texts, train_labels, test_labels = train_test_split(vect_texts, labels, 
                                                                          test_size=0.20, random_state=random_state)
    test_texts, val_texts, test_labels, val_labels = train_test_split(test_texts, test_labels, 
                                                                      test_size=0.50, random_state=random_state)
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def compute_class_weight(labels):
    ''' 
    0: center
    1: left
    2: right
    '''
    samples_per_class = Counter(labels)
    majority_class = max(samples_per_class.values()) 
    class_weight = {0: majority_class/samples_per_class['center'],
                    1: majority_class/samples_per_class['left'],
                    2: majority_class/samples_per_class['right']}
    return class_weight
    
