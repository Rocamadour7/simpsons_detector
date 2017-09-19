import numpy as np


def create_sprite_image(images):
    if isinstance(images, list):
        img_h = images.shape[1]
        img_w = images.shape[2]
        n_plots = int(np.ceil(np.sqrt(images.shape[0])))

        sprite_image = np.ones((img_h * n_plots, img_w * n_plots))

        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < images.shape[0]:
                    this_img = images[this.filter]
                    sprite_image[i * img_h: (i + 1) * img_h: j * img_w: (j + 1) * img_w] = this_img
    return sprite_image


def vector_to_matrix(data, height, width):
    return np.reshape(data, (-1, height, width))