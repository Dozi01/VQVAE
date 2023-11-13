import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vision_utils

device = 'cuda'

def plot_batch(ax, batch, title=None, **kwargs):
    imgs = vision_utils.make_grid(batch, padding=2, normalize=True)
    imgs = np.moveaxis(imgs.numpy(), 0, -1)
    ax.set_axis_off()
    if title is not None: ax.set_title(title)
    return ax.imshow(imgs, **kwargs)

def save_images(batch, title):
    batch_size = batch.shape[0]
    row = int(np.sqrt(batch_size))
    col = batch_size // row
    fig = plt.figure(figsize=(row, col))
    ax = fig.add_subplot(111)
    plot_batch(ax, batch, title)
    file_name = title + '_generated images.png'
    plt.savefig(fname=file_name)