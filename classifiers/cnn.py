from tensorflow.keras.layers import Embedding, Dropout, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D, Dense 
from tensorflow.keras.models import Sequential

class CNN:
    def __init__(self, embedding_layer, num_classes):
        self.embedding_layer = embedding_layer
        self.num_classes = num_classes
    def sepcnn(self, optim, blocks, dropout, filters, kernel_size, pool_size):
        model = Sequential()
        model.add(self.embedding_layer)
        for _ in range(blocks - 1):
            model.add(Dropout(rate=dropout))
            model.add(SeparableConv1D(filters=filters,
                                          kernel_size=kernel_size,
                                          activation='relu',
                                          bias_initializer='random_uniform',
                                          depthwise_initializer='random_uniform',
                                          padding='same'))
            model.add(SeparableConv1D(filters=filters,
                                          kernel_size=kernel_size,
                                          activation='relu',
                                          bias_initializer='random_uniform',
                                          depthwise_initializer='random_uniform',
                                          padding='same'))
            model.add(MaxPooling1D(pool_size=pool_size))

        model.add(SeparableConv1D(filters=filters * 2,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters * 2,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(rate=dropout))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer=optim ,metrics = ['accuracy'])
        model.summary()
        return model