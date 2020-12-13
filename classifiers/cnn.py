from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential

class CNN:
    def __init__(self, embedding_layer, num_classes):
        self.embedding_layer = embedding_layer
        self.num_classes = num_classes
    def toy(self, out_dim):
        model = Sequential()
        model.add(self.embedding_layer)
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model