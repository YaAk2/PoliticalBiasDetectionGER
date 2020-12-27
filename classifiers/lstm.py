from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from classifiers.metrics import*

class LongShortTermMemory:
    def __init__(self, embedding_layer, num_classes):
        self.embedding_layer = embedding_layer
        self.num_classes = num_classes
    def lstm(self, optim, out_dim, dropout):
        model = Sequential()
        model.add(self.embedding_layer)
        model.add(LSTM(out_dim, dropout=dropout))
        model.add(Dense(out_dim/4, activation='elu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer=optim, metrics=['accuracy', f1])
        model.summary()
        return model        
    def stacked_lstm(self, optim, hidden_dim, num_layers, dropout):
        model = Sequential()
        model.add(self.embedding_layer)
        for _ in range(num_layers - 1):
            model.add(LSTM(hidden_dim, return_sequences=True,  dropout=dropout))
        model.add(LSTM(hidden_dim))
        model.add(Dense(hidden_dim/4, activation='elu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer=optim,metrics = ['accuracy', f1])
        model.summary()
        return model