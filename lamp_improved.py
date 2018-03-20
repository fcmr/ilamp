import numpy as np
import lamp

from sklearn import linear_model

from mnist2 import load
from mnist2 import normalize_imgs
from time import time

import matplotlib.pyplot as plt

def plot_embedding(X, lbls, imgs, title=None):
    #x_min, x_max = np.min(X, 0), np.max(X, 0)
    #X = (X - x_min) / (x_max - x_min)

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

    #plt.show()
    plt.savefig(title + '.png')
    plt.clf()



def plot_projection(X, y, title=None, save=False):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    scat = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
    N=10
    bounds = np.linspace(0,N,N+1)
    cb = plt.colorbar(scat, spacing="proportional", ticks=bounds)  
    cb.set_label("Class")

    if title is not None:
        ax1.set_title(title)

    if save == True:
        plt.savefig('ctrl_pts/' + title + '.png')
    else:
        plt.show()

def save_projections(X, num_ctrl_pts, num_neighbors):
    for nctrl in num_ctrl_pts:
        start = time()
        proj = lamp.lamp2d(X_base)
        print("\tlamp projection time: ", time() - start, " - ", nctrl)
        #plot_embedding(proj, y, X_base, "LAMP - {}".format(k))
        plot_projection(proj, y, "LAMP - {}".format(nctrl), save=True)

        for k in num_neighbors:
            start = time()
            lamp.refine_lamp(X_base, proj)
            print("\trefinement time: ", time() - start)
            #plot_embedding(proj, y, X_base, "LAMP Refined {}".format(k))
            plot_projection(proj, y, "LAMP Refined - {} {}".format(nctrl, k), save=True)



# Will only load DATA_SIZE samples
DATA_SIZE = 2000

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

# reshape images into arrays i.e. from a list of images to a list of
# arrays
X_base = np.reshape(X_base, (X_base.shape[0], X_base.shape[1]*X_base.shape[2]))
X_base = X_base[0:DATA_SIZE]
y = y[:DATA_SIZE]


ctrl_pts = [int(np.sqrt(X_base.shape[0])), 100, 150, 200, 250, 300, 350]
num_neighbors = [8, 12, 16, 20]

save_projections(X_base, ctrl_pts, num_neighbors)

#for k in [int(np.sqrt(X_base.shape[0])), 100, 150, 200, 250, 300, 350]:
#    start = time()
#    proj = lamp.lamp2d(X_base)
#    print("\tlamp projection time: ", time() - start, " - ", k)
#    plot_embedding(proj, y, X_base, "LAMP - {}".format(k))
#
#    start = time()
#    lamp.refine_lamp(X_base, proj)
#    print("\trefinement time: ", time() - start)
#    plot_embedding(proj, y, X_base, "LAMP Refined {}".format(k))




#start = time()
#proj = lamp.lamp2d(X_base)
#k = int(np.sqrt(X_base.shape[0]))
#print("\tlamp projection time: ", time() - start, " - ", k)
#plot_embedding(proj, y, X_base, "LAMP Projection - {}".format(k))
#
#
#k = 100
#proj = lamp.lamp2d(X_base, num_ctrl_pts=k)
#print("\tlamp projection time: ", time() - start, " - ", k)
#plot_embedding(proj, y, X_base, "LAMP Projection - {}".format(k))
#
#k = 100
#proj = lamp.lamp2d(X_base, num_ctrl_pts=k)
#print("\tlamp projection time: ", time() - start, " - ", k)
#plot_embedding(proj, y, X_base, "LAMP Projection - {}".format(k))
#
#
#k = 200
#proj = lamp.lamp2d(X_base, num_ctrl_pts=k)
#print("\tlamp projection time: ", time() - start, " - ", k)
#plot_embedding(proj, y, X_base, "LAMP Projection - {}".format(k))
#
#k = 250
#proj = lamp.lamp2d(X_base, num_ctrl_pts=k)
#print("\tlamp projection time: ", time() - start, " - ", k)
#plot_embedding(proj, y, X_base, "LAMP Projection - {}".format(k))
#
#k = 300
#proj = lamp.lamp2d(X_base, num_ctrl_pts=k)
#print("\tlamp projection time: ", time() - start, " - ", k)
#plot_embedding(proj, y, X_base, "LAMP Projection - {}".format(k))
#
#k = 350
#proj = lamp.lamp2d(X_base, num_ctrl_pts=k)
#print("\tlamp projection time: ", time() - start, " - ", k)
#plot_embedding(proj, y, X_base, "LAMP Projection - {}".format(k))

#start = time()
#lamp.refine_lamp(X_base, proj)
#print("\trefinement time: ", time() - start)
#plot_embedding(proj, y, X_base, "LAMP Refined")

#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#print("Computing TSNE embedding")
#start = time()
#X_tsne = tsne.fit_transform(X_base)
#print("\ttsne.fit_transform time: ", time() - start)
#plot_embedding(X_tsne, y, X_base, "tSNE")
