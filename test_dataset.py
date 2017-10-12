import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset
from imgaug import augmenters as iaa

aug = iaa.SomeOf(2, [
    iaa.Affine(rotate=(-10, 10)),
    iaa.Fliplr(1),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AddElementwise((-40, 40)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.5*255), per_channel=0.5)
])
dataset = Dataset(batch_size=16, img_size=128)
inputs, outputs = dataset.next_batch
inputs = aug.augment_images(inputs)
val_X, val_y = dataset.val_set
test_X, test_y = dataset.test_set

## Si quieres ver las imagenes puedes usar Matplotlib
i = 10
plt.imshow(inputs[i]) ## val_X[0], test_X[0]
plt.title(dataset.characters_index[np.argmax(outputs[i])])
plt.show()
