import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation
from dataset import Dataset
import numpy as np

NUM_CLASSES = 20
NUM_EPOCHS = 20
BATCH_SIZE = 256
LR = 0.1
MOMENTUM = 0.9
IMG_SIZE = 64

dataset = Dataset(batch_size=BATCH_SIZE, img_size=IMG_SIZE)


model = Sequential()

model.add(Dense(units=2048, input_dim=IMG_SIZE*IMG_SIZE*3))
model.add(Activation('relu'))
model.add(Dense(units=2048))
model.add(Activation('relu'))
model.add(Dense(units=2048))
model.add(Activation('relu'))
model.add(Dense(units=2048))
model.add(Activation('relu'))
model.add(Dense(units=2048))
model.add(Activation('relu'))
model.add(Dense(units=NUM_CLASSES))
model.add(Activation('softmax'))

optimizer = keras.optimizers.Adam(lr=LR, decay=0.99)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

for epoch_i in range(NUM_EPOCHS):
    dataset.reset_batch_pointer()
    for batch_i in range(dataset.num_batches_in_epoch()):
        batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
        X_batch, y_batch = dataset.next_batch
        X_batch = np.reshape(X_batch, [BATCH_SIZE, IMG_SIZE*IMG_SIZE*3])
        loss, acc = model.train_on_batch(X_batch, y_batch)
        print('{}/{}, epoch: {}, cost: {}, acc: {}'.format(batch_num,
                                                           NUM_EPOCHS * dataset.num_batches_in_epoch(),
                                                           epoch_i, loss, acc))
