import numpy as np
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

TOP_K = 20000
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(tokenized_texts):
    
    #Create vocabulary.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(tokenized_texts)

    #Vectorize data.
    vect_texts = []
    for t in tokenized_texts:
        vect_text = [tokenizer.word_index[w] for w in t]
        vect_texts.append(vect_text)
    vect_texts = np.array(vect_texts, dtype=object)
    
    #Get max sequence length.
    max_length = len(max(vect_texts, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    #Padding 
    vect_texts = sequence.pad_sequences(vect_texts, maxlen=max_length)
    return vect_texts, tokenizer.word_index

def train_val_test_split(vect_texts, labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(vect_texts, onehot_encoded, test_size=0.10)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.10)
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels