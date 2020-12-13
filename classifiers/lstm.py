from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential

class LongShortTermMemory:
    def __init__(self, embedding_layer, num_classes):
        self.embedding_layer = embedding_layer
        self.num_classes = num_classes
    def stacked_lstm(self, optim, out_dim, dropout):
        model = Sequential()
        model.add(self.embedding_layer)
        model.add(LSTM(256, return_sequences=True, dropout=dropout))
        model.add(LSTM(128, return_sequences=True, dropout=dropout))
        model.add(LSTM(64, return_sequences=True, dropout=dropout))
        model.add(LSTM(out_dim, dropout=dropout))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer=optim,metrics = ['accuracy'])
        model.summary()
        return model