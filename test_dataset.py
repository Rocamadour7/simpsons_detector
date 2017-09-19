import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset

dataset = Dataset(batch_size=16, img_size=128)
inputs, outputs = dataset.next_batch()
val_X, val_y = dataset.val_set
test_X, test_y = dataset.test_set

## Si quieres ver las imagenes puedes usar Matplotlib
i = 10
plt.imshow(inputs[i]) ## val_X[0], test_X[0]
plt.title(dataset.characters_index[np.argmax(outputs[i])])