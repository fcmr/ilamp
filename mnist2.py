import os
import sys
import struct
import numpy as np

import warnings

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from time import time
from skimage.transform import resize
from skimage import img_as_float

import lamp

"""
Basically copied from: https://gist.github.com/akesling/5358964
"""
def load(dataset='train', path='data/'):
    # http://yann.lecun.com/exdb/mnist/
    if dataset is 'train':
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is 'test':
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        sys.exit('Error: invalid dataset')


    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8)
        img = img.reshape(len(lbl), rows, cols)

    return img, lbl


def plot_embedding(X, lbls, imgs, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(lbls[i]),
                 color=plt.cm.Set1(lbls[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    #if hasattr(offsetbox, 'AnnotationBbox'):
    #    # only print thumbnails with matplotlib > 1.0
    #    shown_images = np.array([[1., 1.]])  # just something big
    #    for i in range(imgs.shape[0]):
    #        dist = np.sum((X[i] - shown_images) ** 2, 1)
    #        if np.min(dist) < 4e-3:
    #            # don't show points that are too close
    #            continue
    #        shown_images = np.r_[shown_images, [X[i]]]
    #        img_sz = int(np.sqrt(imgs[0].shape[0]))
    #        imagebox = offsetbox.AnnotationBbox(
    #            offsetbox.OffsetImage(imgs[i].reshape((img_sz, img_sz)), cmap=plt.cm.gray_r),
    #            X[i])
    #        ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.show()
    #plt.savefig(title + '.png')
    #plt.clf()


# resizes all images in orig to new_shape
def resize_imgs(orig, new_shape):
    new = []
    for sample in orig:
        new.append(resize(sample, new_shape))
    return np.array(new)

def normalize_imgs(orig):
    new = []
    for sample in orig:
        new.append(img_as_float(sample))
    return np.array(new)


# creates a squate of (n_imgs x n_imgs) samples from the input imgs then
# plots this square
def plot_grid_samples(imgs, n_imgs):
    img_sz = imgs.shape[1]
    n_img_per_row = 20
    img = np.zeros(((img_sz + 2)*n_imgs, (img_sz + 2)* n_imgs))
    for i in range(n_imgs):
        ix = (img_sz + 2)*i + 1
        for j in range(n_imgs):
            iy = (img_sz + 2)*j + 1
            img[ix:ix + img_sz, iy:iy + img_sz] = imgs[i * n_img_per_row + j]

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('Digits')
    plt.show()

warnings.filterwarnings("ignore")

# Will only load DATA_SIZE samples
DATA_SIZE = 10000

print("Loading datset")
start = time()
# TODO: shuffle data
X_orig, y = load()
print("\tLoading time: ", time() - start)

# TODO: resize? 
# tsne on 28x28 images is too slow and returns bad embeddings, test on LAMP
print("Normalizing...")
start = time()
X_base = normalize_imgs(X_orig)
print("\tNormalizing time: ", time() - start)
#plot_grid_samples(X_base, 20)

# reshape images into arrays i.e. from a list of images to a list of
# arrays
X_base = np.reshape(X_base, (X_base.shape[0], X_base.shape[1]*X_base.shape[2]))
X_base = X_base[0:DATA_SIZE]

start = time()
proj = lamp.lamp2d(X_base)
print("\tlamp projection time: ", time() - start)

plot_embedding(proj, y, X_base, "LAMP Projection")

# propose new samples with iLAMP
new_img = lamp.ilamp(X_base, proj, np.array([0.24,0.23]))
plt.imshow(new_img.reshape((28, 28)))
plt.show()

new_img = lamp.ilamp(X_base, proj, np.array([0.95,0.72]))
plt.imshow(new_img.reshape((28, 28)))
plt.show()

new_img = lamp.ilamp(X_base, proj, np.array([0.13 ,0.8]))
plt.imshow(new_img.reshape((28, 28)))
plt.show()

new_img = lamp.ilamp(X_base, proj, np.array([0.95, 0.58]))
plt.imshow(new_img.reshape((28, 28)))
plt.show()
