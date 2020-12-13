import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

MAX_FEATURES = 20000
MAX_SEQUENCE_LENGTH = 500

class SequenceVectorize:
    def __init__(self, tokenized_texts, vocab=False):
        '''
        vocab: Boolean, if True we calculate the vocabulary from the dataset.
        '''
        self.tokenized_texts = tokenized_texts
            
        #Get vocabulary.
        if vocab==False:
            with open("GloVe/vocab.txt") as f:
                self.word_index = {}
                i = 1
                for line in f:
                    (key, _) = line.split()
                    self.word_index[key] = i
                    i+=1
                    if i == MAX_FEATURES + 1:
                        break
        else:
            tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
            tokenizer.fit_on_texts(tokenized_texts)
            self.word_index = tokenizer.word_index

    def vectorize(self):
        vect_texts = []
        for t in self.tokenized_texts:
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
        vect_texts = sequence.pad_sequences(vect_texts, maxlen=MAX_SEQUENCE_LENGTH)
        return vect_texts
    
def onehot_encoding(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(labels), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded, label_encoder
    
def onehot_decoding(onehot_encoded, label_encoder):
    onehot_decoded = label_encoder.inverse_transform([np.argmax(onehot_encoded)])
    return onehot_decoded

def train_val_test_split(vect_texts, labels, reproduceable=True):
    '''
    reproduceable: Boolean, if True we always get exactly the same shuffling
    '''
    if reproduceable == True:
        random_state = 20
    else:
        random_state = None
    
    onehot_encoded = onehot_encoding(labels)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(vect_texts, onehot_encoded, 
                                                                          test_size=0.20, random_state=random_state)
    test_texts, val_texts, test_labels, val_labels = train_test_split(test_texts, test_labels, 
                                                                      test_size=0.50, random_state=random_state)
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels