import numpy as np
from tensorflow.keras.layers import Embedding

def get_embedding(word_index, max_sequence_length, pretrained_embedding, finetune_embedding, embedding_dim=None):
    '''
    pretrained_embedding: Boolean, if True we take a pretrained embedding.
    finetune_embedding: Boolean, if True we train the pretrained-embedding.
    embedding_dim: Integer, Needs to be specified if pretrained_embedding is False.
    '''
    max_features = len(word_index)
    if pretrained_embedding:
        embeddings_index = {}
        f = open('PretrainedEmbedding/vectors.txt')
        i = 1
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            i+=1
            if i == max_features + 1:
                break
        
        embedding_dim = len(embeddings_index[word])
        embedding_matrix = np.zeros((max_features + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(max_features + 1,
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=finetune_embedding)
    else:
        embedding_layer = Embedding(max_features + 1,
                                    embedding_dim,
                                    input_length=max_sequence_length)

    return embedding_layer